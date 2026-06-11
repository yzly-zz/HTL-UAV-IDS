"""桩脚本：源域预训练（统一入口）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.training.pretrain import run_pretrain

if __name__ == "__main__":
    run_pretrain("data/raw/CIC-ToN-IoT-V2.parquet")