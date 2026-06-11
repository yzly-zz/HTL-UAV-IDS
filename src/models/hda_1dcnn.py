import torch
import torch.nn as nn

"""
    这是论文的核心算法载体。有一个非常关键的函数 freeze_backbone_for_finetuning()，
    这将在写论文论述“如何避免灾难性遗忘 (Catastrophic Forgetting)”时提供直接的代码支撑。
"""

class DepthwiseSeparableConv1d(nn.Module):
    """
    深度可分离一维卷积模块
    学术价值：通过将标准卷积拆分为深度卷积(Depthwise)和逐点卷积(Pointwise)，
    在极低精度损失的前提下，大幅降低边缘设备的浮点运算量(FLOPs)和参数量。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class DomainProjector(nn.Module):
    """
    领域特异性投影层 (Domain-Specific Projector)
    学术价值：解决 ToN-IoT 和 UAV-NIDD 异构维度不匹配的核心组件。
    它将任意输入维度映射到统一的公共语义潜空间 (Shared Latent Space)。
    """

    def __init__(self, input_dim, shared_dim=64):
        super(DomainProjector, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.GELU()  # 替换 ReLU 为 GELU，梯度更平滑，有利于潜空间对齐
        )

    def forward(self, x):
        return self.projection(x)


class ProjBridge(nn.Module):
    """
    ProjBridge: Projection-based Heterogeneous Transfer for UAV Intrusion Detection
    完整架构 = 动态投影层 + 共享主干网络 + 分类头
    """

    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super(ProjBridge, self).__init__()

        # 1. 动态投影层：在微调阶段会被替换
        self.projector = DomainProjector(input_dim, shared_dim)

        # 2. 共享特征提取主干 (Shared Backbone)
        # 注意：这里的 Sequence Length 固定为 shared_dim
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            DepthwiseSeparableConv1d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2),
            DepthwiseSeparableConv1d(32, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化，彻底消除对序列长度的依赖
        )

        # 3. 分类头 (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止小样本目标域微调时过拟合
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Input_Dim)

        # 阶段一：异构维度对齐
        projected_x = self.projector(x)  # (Batch, shared_dim)

        # 阶段二：升维以适配 1D-CNN (Batch, Channels=1, Sequence=shared_dim)
        cnn_input = projected_x.unsqueeze(1)

        # 阶段三：共享特征提取
        features = self.backbone(cnn_input)  # (Batch, 64, 1)
        features = features.squeeze(-1)  # (Batch, 64)

        # 阶段四：分类
        logits = self.classifier(features)  # (Batch, num_classes)
        return logits

    def freeze_backbone_for_finetuning(self):
        """
        冻结主干网络策略 (学术论点核心)
        在目标域微调时调用此方法。强制冻结泛化能力强的 backbone，
        仅开放 projector 和 classifier 进行梯度更新。
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[模型架构] 主干网络 (Backbone) 已冻结。仅训练 Projector 和 Classifier。")


# Legacy compatibility alias for old scripts. New code should use ProjBridge directly.
HTL_UAV_IDS = ProjBridge
