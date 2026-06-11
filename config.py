from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_SEED = 42
SHARED_DIM = 64

SOURCE_VAL_RATIO = 0.2
TARGET_VAL_RATIO = 0.2
TARGET_TEST_RATIO = 0.2

SOURCE_BATCH_SIZE = 512
TARGET_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

SOURCE_MODEL_PATH = WEIGHTS_DIR / "source_base_model.pth"

SCENARIOS = {
    "uav": {
        "name": "UAV-Case1（无人机节点 / WiFi 帧层）",
        "files": [DATA_DIR / "UAV-NIDD" / "UAV-Case1-Label.csv"],
        "label_col": None,
        "default_class_weight": None,
        "split_mode": "block_stratified",
        "split_column": "frame.number",
        "block_size": 5000,
    },
    "ap": {
        "name": "Access Point Case2（接入点 / 混合层）",
        "files": [DATA_DIR / "UAV-NIDD" / "Access Point Case2 Label.csv"],
        "label_col": None,
        "default_class_weight": None,
        "split_mode": "block_stratified",
        "split_column": "frame.number",
        "block_size": 5000,
    },
    "gcs": {
        "name": "GCS Case3（地面控制站 / 流统计层）",
        "files": [DATA_DIR / "UAV-NIDD" / "GSC Case3 Label.csv"],
        "label_col": None,
        "default_class_weight": None,
        "split_mode": "random_stratified",
        "split_column": None,
        "block_size": None,
    },
}
