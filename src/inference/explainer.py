"""
功能描述：
本脚本实现了针对无人机边缘计算优化的轻量级特征归因算法 (SHAP-Lite)。
核心机制：使用 Gradient * Input (输入特征值乘以其预测类别的偏导数) 来近似特征重要性。
数学表达：$Attribution_i = x_i \times \frac{\partial y_c}{\partial x_i}$
学术价值：规避了传统基于扰动 (Perturbation-based) 的 SHAP 或 LIME 极高的采样算力开销，
实现微秒级的异常溯源，完美契合资源受限的边缘环境。
"""

import torch
import numpy as np


class EdgeExplainer:
    def __init__(self, model, feature_names=None, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.feature_names = feature_names

    def explain_instance(self, input_tensor, top_k=5):
        """
        计算单条流量特征的归因贡献度
        :param input_tensor: 形状为 (1, D_tgt) 的张量
        :return: 贡献度排名前 K 的特征及其得分
        """
        self.model.eval()

        # 必须克隆张量并允许求导
        x = input_tensor.clone().detach().to(self.device)
        x.requires_grad = True

        # 前向传播
        logits = self.model(x)
        pred_class = torch.argmax(logits, dim=1).item()

        # 获取预测类别的分值并反向传播
        target_score = logits[0, pred_class]
        self.model.zero_grad()
        target_score.backward()

        # 计算 Gradient * Input (归因分数)
        gradients = x.grad.data
        # 新（修复）:
        attribution_tensor = (gradients * x).squeeze(0)
        # 从计算图分离并移动到 CPU，再转为 numpy
        attributions = attribution_tensor.detach().cpu().numpy()

        # 取绝对值作为重要性指标，并归一化为百分比
        importance = np.abs(attributions)
        sum_importance = np.sum(importance) + 1e-9  # 防止除以0
        normalized_importance = (importance / sum_importance) * 100

        # 获取 Top-K 索引
        top_indices = np.argsort(normalized_importance)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            feat_name = self.feature_names[idx] if self.feature_names is not None else f"Feature_{idx}"
            results.append({
                "feature_index": int(idx),
                "feature_name": feat_name,
                "contribution_pct": float(normalized_importance[idx])
            })

        return results

    def format_audit_report(self, explain_results):
        """格式化输出可信审计日志，直接推送给地面站大屏"""
        lines = ["[边缘审计溯源] 异常流量拦截报告"]
        lines.append("发现恶意入侵信号！关键致因特征 (Top贡献度):")
        for i, res in enumerate(explain_results):
            lines.append(f"  {i + 1}. {res['feature_name']} -> 威胁贡献比: {res['contribution_pct']:.1f}%")
        return "\n".join(lines)