"""
V3 边缘推理基准测试脚本
测量 ProjBridge 及所有基线和消融变体在 GPU/CPU 上的：
  - 推理延迟 (ms/sample)
  - 吞吐量 (samples/sec)
  - 参数量 (可训练 / 总量)
  - 模型文件大小 / 内存占用

用法：
  python scripts/benchmark_edge_v3.py --device cuda
  python scripts/benchmark_edge_v3.py --device cpu
"""

import torch
import torch.nn as nn
import time
import numpy as np
import os, sys, json, joblib, argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hda_1dcnn import ProjBridge, DomainProjector, DepthwiseSeparableConv1d

# ── 场景维度映射（与 finetune.py 训练产物一致） ──
SCENARIO_DIMS = {
    "uav": 28,
    "ap": 29,
    "gcs": 44,
}

# ── 所有待测试模型 ──
class VanillaMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.net(x)


class Standard1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.net(x).squeeze(-1))


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class MobileNet1D(nn.Module):
    def __init__(self, input_dim, num_classes=2, shared_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            DepthwiseSeparableConv1d(1, 16, 3, 1), nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 32, 3, 1), nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(32, 64, 3, 1), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


class LightTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])


# ── 消融变体 ──
class ProjBridge_NoProj(nn.Module):
    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super().__init__()
        self.shared_dim = shared_dim
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 32, 3, 1), nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(32, 64, 3, 1), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, x):
        if x.shape[1] < self.shared_dim:
            x = torch.cat([x, torch.zeros(x.shape[0], self.shared_dim - x.shape[1], device=x.device)], dim=1)
        elif x.shape[1] > self.shared_dim:
            x = x[:, :self.shared_dim]
        return self.classifier(self.backbone(x.unsqueeze(1)).squeeze(-1))


class ProjBridge_StdConv(nn.Module):
    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super().__init__()
        self.projector = DomainProjector(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.projector(x).unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


def benchmark_model(model, input_dim, device, n_warmup=100, n_test=1000):
    """测量单个模型的推理延迟和吞吐量"""
    model = model.to(device).eval()
    x = torch.randn(1, input_dim).to(device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # 测量延迟
    latencies = []
    with torch.no_grad():
        for _ in range(n_test):
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    avg_latency = np.mean(latencies)
    throughput = 1000 / avg_latency  # samples/sec

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 内存估算（参数 × 4 bytes float32）
    memory_mb = total_params * 4 / (1024 * 1024)

    return {
        "avg_latency_ms": round(avg_latency, 4),
        "throughput_sps": round(throughput, 1),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "memory_mb": round(memory_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    # ── 场景：使用 GCS 维度（最难场景的模型结构最完整） ──
    input_dim = 44

    # ── 定义所有待测试模型 ──
    models_to_test = {
        "ProjBridge (Ours)":     ProjBridge(input_dim=input_dim),
        "MLP":                    VanillaMLP(input_dim=input_dim),
        "Standard 1D-CNN":        Standard1DCNN(input_dim=input_dim),
        "LSTM":                   LSTMBaseline(input_dim=input_dim),
        "MobileNet1D":            MobileNet1D(input_dim=input_dim),
        "LightTransformer":       LightTransformer(input_dim=input_dim),
        "ProjBridge-noProj":      ProjBridge_NoProj(input_dim=input_dim),
        "ProjBridge-stdConv":     ProjBridge_StdConv(input_dim=input_dim),
    }

    results = []
    print(f"{'Model':<25s} {'延迟(ms)':>9s} {'吞吐(s/s)':>10s} {'参数量':>8s} {'内存(MB)':>9s}")
    print("-" * 70)

    for name, model in models_to_test.items():
        r = benchmark_model(model, input_dim, device)
        results.append({"model": name, **r})
        print(f"{name:<25s} {r['avg_latency_ms']:>8.4f} {r['throughput_sps']:>9.0f} "
              f"{r['total_params']:>7,} {r['memory_mb']:>8.2f}")

    # ── 保存为 CSV ──
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = f"results/edge_benchmark_{args.device}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 已保存: {csv_path}")


if __name__ == "__main__":
    main()