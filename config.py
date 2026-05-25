import torch
import os

# -------------------- 项目根目录 --------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- 数据路径 --------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")

SOURCE_FILE = os.path.join(RAW_DIR, "CIC-ToN-IoT-V2.parquet")
UAV_FILES = [
    os.path.join(RAW_DIR, "UAV-NIDD", "Access Point Case2 Label.csv"),
    os.path.join(RAW_DIR, "UAV-NIDD", "GSC Case3 Label.csv"),
    os.path.join(RAW_DIR, "UAV-NIDD", "UAV-Case1-Label.csv"),
]

# -------------------- 权重与结果保存 --------------------
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- 模型命名（统一标准） --------------------
SOURCE_PRETRAIN_NAME = "source_pretrained"
TARGET_MODEL_NAME = "HTL-UAV-IDS"            # 最终模型
ABLATION_NAMES = {
    "HTL-noProj": "HTL-noProj",
    "HTL-noFreeze": "HTL-noFreeze",
    "HTL-stdConv": "HTL-stdConv",
    "HTL-Scratch": "HTL-Scratch",
}
BASELINE_DL_NAMES = {
    "MLP": "MLP_baseline",
    "Standard1DCNN": "Standard1DCNN_baseline",
    "LSTM": "LSTM_baseline",
    "MobileNet1D": "MobileNet1D_baseline",
    "LightTransformer": "LightTransformer_baseline",
}

# -------------------- 设备与超参数 --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
SHARED_DIM = 64
LR = 0.002
EPOCHS = 30                       # 从零训练轮数
FINETUNE_LR = 0.0002
FINETUNE_EPOCHS = 20              # 微调轮数
RANDOM_SEED = 42                  # 全局主种子

# -------------------- 样本效率实验简化参数 --------------------
SAMPLE_EFFICIENCY_RATIOS = [0.05, 0.1, 0.25, 0.5, 1.0]
SAMPLE_EFFICIENCY_SEEDS = [42]           # 仅用1个种子
SAMPLE_EFFICIENCY_EPOCHS_OURS = 15            # 缩短训练
SAMPLE_EFFICIENCY_EPOCHS_BASELINE = 20

# ---------- 改进策略超参数 ----------
FOCAL_GAMMA = 2.0               # Focal Loss 聚焦参数
MIXUP_ALPHA = 0.2               # Mixup 强度
BACKBONE_LR = 1e-5              # 解冻 backbone 后的极低学习率
PROJECTOR_LR = 1e-4             # 投影层学习率
HEAD_LR = 1e-4                  # 分类头学习率
T_MAX = FINETUNE_EPOCHS         # 余弦退火周期