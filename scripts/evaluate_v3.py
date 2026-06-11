import argparse
import json
import os
import sys

import joblib
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, SCENARIOS, SHARED_DIM, WEIGHTS_DIR
from src.models.hda_1dcnn import ProjBridge
from src.training.experiment_utils import (
    evaluate_classifier,
    load_and_clean,
    load_split_indices,
    make_dataloader,
    slice_splits,
)


def evaluate_scenario(scenario_key, device="cuda"):
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key

    print(f"\n{'=' * 60}")
    print(f"  评估: {scenario['name']}")
    print(f"{'=' * 60}")

    features, labels, _ = load_and_clean(scenario["files"], label_col=scenario["label_col"])
    split_path = WEIGHTS_DIR / f"target_split_{suffix}.npz"
    scaler_path = WEIGHTS_DIR / f"target_scaler_{suffix}.pkl"
    model_path = WEIGHTS_DIR / f"target_uav_model_{suffix}.pth"

    if not split_path.exists():
        raise FileNotFoundError(f"缺少 split 文件: {split_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"缺少 scaler 文件: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"缺少模型文件: {model_path}")

    split_indices = load_split_indices(split_path)
    split_data = slice_splits(features, labels, split_indices)
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(split_data["test"][0])
    y_test = split_data["test"][1]

    test_loader = make_dataloader(X_test, y_test, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    model = ProjBridge(input_dim=features.shape[1], shared_dim=SHARED_DIM, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    metrics = evaluate_classifier(model, test_loader, device=device)
    cm = metrics["confusion_matrix"]
    report = metrics["classification_report"]

    print(f"  Test Acc: {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']}")
    print(f"  PR-AUC: {metrics['pr_auc']}")
    print("  Classification Report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  Confusion Matrix: {cm}")

    return {
        "scenario": scenario_key,
        "name": scenario["name"],
        "accuracy": metrics["accuracy"],
        "auc_roc": metrics["auc_roc"],
        "pr_auc": metrics["pr_auc"],
        "confusion_matrix": cm,
        "test_samples": int(len(y_test)),
        "attack_rate": float(y_test.mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}
    for key in ["uav", "ap", "gcs"]:
        try:
            results[key] = evaluate_scenario(key, device=args.device)
        except FileNotFoundError as exc:
            print(exc)

    print(f"\n{'=' * 70}")
    print("  三场景综合评估汇总")
    print(f"{'=' * 70}")
    for key in ["uav", "ap", "gcs"]:
        if key not in results:
            continue
        record = results[key]
        print(
            f"  {record['name']}: "
            f"Acc={record['accuracy']:.4f}, "
            f"AUC={record['auc_roc']}, "
            f"PR-AUC={record['pr_auc']}, "
            f"Test={record['test_samples']}"
        )
