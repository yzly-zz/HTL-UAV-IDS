"""桩脚本：目标域微调（统一入口）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
from src.training.finetune import run_finetune, SCENARIOS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            run_finetune(key, epochs=args.epochs, lr=args.lr, device=args.device)
    else:
        run_finetune(args.scenario, epochs=args.epochs, lr=args.lr, device=args.device)