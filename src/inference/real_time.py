"""
功能描述：
边缘节点异步实时推理引擎。
学术价值：通过环形队列 (Queue) 实现网卡数据捕获与 GPU/CPU 推理的多线程解耦，
保证突发性洪水攻击 (DDoS/Flood) 下系统不会因为 I/O 阻塞而崩溃。
当检测到攻击时，调用轻量级 Explainer 进行归因，并通过 HTTP 异步推送到 GCS。
"""
import torch
import time
import threading
import queue
import requests
import json
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.hda_1dcnn import ProjBridge
from src.inference.explainer import EdgeExplainer
import joblib


class UAVEdgeInferenceEngine:
    def __init__(self, target_weights_path, target_scaler_path, input_dim, gcs_api_url, device='cpu'):
        self.device = torch.device(device)
        self.gcs_api_url = gcs_api_url
        self.input_dim = input_dim

        print(f"[Edge Engine] 初始化无人机推理节点 | 设备: {self.device}")

        # 1. 加载目标域（UAV）微调后的完整异构模型
        # 先尝试加载 scaler，以从 scaler 推断正确的特征维度
        try:
            self.scaler = joblib.load(target_scaler_path)
        except Exception as e:
            raise RuntimeError(f"[Edge Engine] 无法加载 scaler: {target_scaler_path}. 错误: {e}")

        # 如果 scaler 可用且包含 mean_，以 scaler.mean_.shape[0] 作为准则
        scaler_dim = None
        if hasattr(self.scaler, "mean_") and getattr(self.scaler, "mean_") is not None:
            scaler_dim = int(getattr(self.scaler, "mean_").shape[0])

        if scaler_dim is not None and scaler_dim != input_dim:
            print(
                f"[Edge Engine] 检测到传入 input_dim={input_dim} 与 scaler 期望维度 {scaler_dim} 不一致，已改用 scaler 的维度。")
            self.input_dim = scaler_dim
        else:
            self.input_dim = input_dim

        # 基于 self.input_dim 创建模型并加载权重（保持严格加载，保证结构完全一致）
        self.model = ProjBridge(input_dim=self.input_dim, shared_dim=64, num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(target_weights_path, map_location=self.device))
        self.model.eval()

        # 3. 初始化可解释性引擎 (SHAP-Lite)
        self.explainer = EdgeExplainer(self.model, device=self.device)

        # 4. 异步并发队列
        self.traffic_queue = queue.Queue(maxsize=5000)
        self.is_running = True

    def _simulated_packet_capture(self, mock_data_generator):
        """
        生产者线程：模拟高速网卡抓包 (真实部署中替换为 libpcap/socket 逻辑)
        """
        while self.is_running:
            try:
                # 模拟获取一行原生数据 (未标准化的 Numpy Array)
                raw_flow = next(mock_data_generator)
                if not self.traffic_queue.full():
                    self.traffic_queue.put(raw_flow)
            except StopIteration:
                self.is_running = False
                break
            time.sleep(0.01)  # 模拟真实网络包间隔

    def _inference_worker(self):
        """
        消费者线程：负责高速弹出队列数据、归一化、推理及告警上报
        """
        while self.is_running or not self.traffic_queue.empty():
            try:
                # 设置 100ms 超时，防止线程死锁
                raw_flow = self.traffic_queue.get(timeout=0.1)

                start_time = time.perf_counter()

                # 1. 特征标准化 (必须转换为 2D 进行 transform，再转回 1D)
                scaled_flow = self.scaler.transform(raw_flow.reshape(1, -1))
                input_tensor = torch.tensor(scaled_flow, dtype=torch.float32).to(self.device)

                # 2. 前向推理
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    pred = torch.argmax(logits, dim=1).item()

                latency_ms = (time.perf_counter() - start_time) * 1000

                # 3. 告警与可解释性审计逻辑
                is_attack = bool(pred == 1)
                explanation_text = "正常流量，无异常溯源。"

                if is_attack:
                    try:
                        explain_results = self.explainer.explain_instance(input_tensor)
                        explanation_text = self.explainer.format_audit_report(explain_results)
                        print(f"\n[🚨 拦截告警] 推理耗时: {latency_ms:.2f}ms\n{explanation_text}")
                    except Exception as e:
                        # 记录异常但不让线程崩溃
                        print(f"[WARN] Explainer 归因失败: {e}")
                        explanation_text = "可解释性失败：无法生成解释"

                # 4. 推送到 GCS 地面站 (由于 requests.post 会阻塞，严谨的工业代码应将其放入另一个线程或使用 aiohttp，此处从简)
                payload = {
                    "timestamp": time.time(),
                    "latency_ms": latency_ms,
                    "is_attack": is_attack,
                    "explanation": explanation_text
                }

                try:
                    requests.post(self.gcs_api_url, json=payload, timeout=0.5)
                except requests.RequestException:
                    pass  # 忽略网络断开，保证边缘节点自身稳定运行

                self.traffic_queue.task_done()

            except queue.Empty:
                continue

    def start_engine(self, mock_data_generator):
        print("[Edge Engine] 引擎启动，多线程网卡监听与推理流水线已激活。")
        producer = threading.Thread(target=self._simulated_packet_capture, args=(mock_data_generator,))
        consumer = threading.Thread(target=self._inference_worker)

        producer.start()
        consumer.start()

        producer.join()
        consumer.join()
        print("[Edge Engine] 引擎安全关闭。")


# 测试启动代码
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="gcs",
                        choices=["uav", "ap", "gcs"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--attack_rate", type=float, default=0.15,
                        help="模拟攻击流量比例 (0.0~1.0)")
    args = parser.parse_args()

    suffix = args.scenario
    weights_path = f"weights/target_uav_model_{suffix}.pth"
    scaler_path  = f"weights/target_scaler_{suffix}.pkl"
    feature_path = f"weights/target_feature_names_{suffix}.json"

    # ── 动态推断特征维度 ──
    feat_dim = 44
    if os.path.exists(feature_path):
        with open(feature_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        feat_dim = len(feature_names)
        print(f"[Edge Engine] Scenario={suffix}, dim={feat_dim}")
    else:
        print(f"[WARN] 未找到 {feature_path}，使用 fallback dim=44")

    # ── 从真实 CSV 加载样本，构建模拟数据池 ──
    csv_path = f"data/raw/UAV-NIDD/GSC Case3 Label.csv"
    normal_samples = []
    attack_samples = []

    if os.path.exists(csv_path):
        print(f"[Edge Engine] 从真实数据构建模拟流: {csv_path}")
        for enc in ["utf-8", "gbk", "latin-1", "ISO-8859-1", "cp1252"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        # 使用 preprocessor 清洗数据，获取数值特征矩阵和标签
        from src.data_engine.preprocessor import HeterogeneousDataPreprocessor
        prep = HeterogeneousDataPreprocessor()
        X_raw, y, _ = prep._clean_dataframe(df, label_col=None)

        for i in range(len(y)):
            if y[i] == 0:
                normal_samples.append(X_raw[i])
            else:
                attack_samples.append(X_raw[i])

        print(f"[Edge Engine] 正常样本池: {len(normal_samples)}, 攻击样本池: {len(attack_samples)}")
    else:
        print(f"[WARN] 未找到 {csv_path}，将使用随机噪声作为模拟数据")

    # ── 模拟数据生成器 ──
    def dummy_data_gen():
        import random
        has_real_data = len(normal_samples) > 0 and len(attack_samples) > 0

        while True:
            if has_real_data:
                if random.random() < args.attack_rate:
                    # 随机选取一个攻击样本
                    idx = random.randint(0, len(attack_samples) - 1)
                    raw_flow = attack_samples[idx].astype(np.float64)
                else:
                    # 随机选取一个正常样本
                    idx = random.randint(0, len(normal_samples) - 1)
                    raw_flow = normal_samples[idx].astype(np.float64)
            else:
                raw_flow = np.random.rand(feat_dim)

            yield raw_flow
            time.sleep(0.05)  # 模拟真实网络包间隔（降低间隔以提高数据速率）

    engine = UAVEdgeInferenceEngine(
        target_weights_path=weights_path,
        target_scaler_path=scaler_path,
        input_dim=feat_dim,
        gcs_api_url="http://127.0.0.1:8000/api/report_traffic",
        device=args.device,
    )
    engine.start_engine(dummy_data_gen())