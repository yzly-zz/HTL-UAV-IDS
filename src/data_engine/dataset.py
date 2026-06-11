import torch
from torch.utils.data import Dataset
import numpy as np

"""
    为了让底层的 NumPy 数组能够高效地喂给 GPU 进行神经网络训练，
    需要将其包装为标准的 PyTorch Dataset，这是打通数据预处理和模型训练的关键桥梁。
"""


class IDSStreamDataset(Dataset):
    """
    轻量级入侵检测数据集流
    将预处理后的 Numpy 数据转换为 PyTorch Tensor，支持 DataLoader 进行批量和打乱操作。
    """

    def __init__(self, features, labels):
        """
        :param features: Numpy array 格式的特征矩阵 (N, D) 或 不规则的特征列表
        :param labels: Numpy array 格式的标签向量 (N,)
        """
        # 【修改重点：学术级无损维度对齐 (Zero-Padding)】
        # 绝不丢弃任何数据！如果遇到 43 维和 44 维混杂的情况，通过末尾补 0 强制对齐。
        if isinstance(features, list) or (isinstance(features, np.ndarray) and features.dtype == object):
            print("\n[Dataset] 🛡️ 检测到维度不齐，启动无损 Zero-Padding 对齐机制...")
            # 找到当前数据集中的最大特征维度（例如 44）
            lengths = [len(f) for f in features]
            max_dim = max(lengths)

            padded_features = []
            for f in features:
                if len(f) < max_dim:
                    # 如果维度不足，在末尾填充 0，补齐至 max_dim
                    pad_width = max_dim - len(f)
                    padded_f = np.pad(f, (0, pad_width), 'constant', constant_values=0)
                    padded_features.append(padded_f)
                else:
                    padded_features.append(f)

            # 将填充好的数据重新转换为规整的 Numpy 矩阵
            features = np.array(padded_features)
            print(f"[Dataset] ✅ 无损对齐完成！全部数据已对齐至 {max_dim} 维，未丢弃任何样本。")

        # 为了兼容后续 1D-CNN 和全连接层，统一使用 Float32 类型
        self.X = torch.tensor(features, dtype=torch.float32)
        # 交叉熵损失函数要求标签类型为 Long
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]