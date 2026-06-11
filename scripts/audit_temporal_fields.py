import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, SCENARIOS, SHARED_DIM, SOURCE_MODEL_PATH, TARGET_BATCH_SIZE
from src.models.hda_1dcnn import ProjBridge
from src.training.experiment_utils import (
    evaluate_classifier,
    load_dataframe_and_clean,
    load_pretrained_backbone,
    make_block_split_indices,
    make_dataloader,
    make_split_indices,
    run_training_loop,
    scale_split_data,
    set_seed,
    slice_splits,
)


SUSPICIOUS_PATTERNS = {
    "uav": [
        "frame.number",
        "frame.time_delta_displayed",
        "frame.time_epoch",
        "frame.time_relative",
        "radiotap.timestamp.ts",
        "radiotap.mactime",
        "wlan_radio.timestamp",
        "wlan_radio.start_tsf",
    ],
    "ap": [
        "frame.number",
        "frame.time_delta_displayed",
        "frame.time_epoch",
        "frame.time_relative",
        "radiotap.timestamp.ts",
        "radiotap.mactime",
        "wlan_radio.end_tsf",
    ],
}


def _split_for_scenario(dataframe, labels, scenario, seed):
    if scenario.get("split_mode") == "block_stratified":
        return make_block_split_indices(
            dataframe=dataframe,
            labels=labels,
            split_column=scenario["split_column"],
            block_size=scenario["block_size"],
            seed=seed,
        )
    return make_split_indices(labels, seed=seed)


def _train_projbridge(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, device, epochs, lr):
    train_loader = make_dataloader(x_train, y_train, batch_size=TARGET_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(x_test, y_test, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    model = ProjBridge(input_dim=input_dim, shared_dim=SHARED_DIM, num_classes=2).to(device)
    if load_pretrained_backbone(model, SOURCE_MODEL_PATH, device=device):
        model.freeze_backbone_for_finetuning()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc, _ = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )
    metrics = evaluate_classifier(model, test_loader, device=device, criterion=criterion)
    report = metrics["classification_report"]
    return {
        "best_val_acc": best_val_acc,
        "test_acc": metrics["accuracy"],
        "test_auc_roc": metrics["auc_roc"],
        "f1_macro": report.get("macro avg", {}).get("f1-score"),
        "f1_normal": report.get("Normal", {}).get("f1-score"),
        "f1_attack": report.get("Attack", {}).get("f1-score"),
    }


def run_audit(scenario_key, epochs=10, lr=1e-4, device="cuda", seed=DEFAULT_SEED):
    if scenario_key not in SUSPICIOUS_PATTERNS:
        raise ValueError(f"Temporal-field audit is only defined for: {list(SUSPICIOUS_PATTERNS)}")

    set_seed(seed)
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    patterns = SUSPICIOUS_PATTERNS[scenario_key]

    dataframe, features, labels, feature_names = load_dataframe_and_clean(
        scenario["files"],
        label_col=scenario["label_col"],
    )
    split_indices = _split_for_scenario(dataframe, labels, scenario, seed)

    numeric_cols = [col for col in feature_names]
    drop_cols = [col for col in numeric_cols if any(pattern in col for pattern in patterns)]
    keep_cols = [col for col in numeric_cols if col not in drop_cols]

    print(f"\n{'=' * 60}")
    print(f"  Temporal field audit: {scenario['name']}")
    print(f"  Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"{'=' * 60}")
    print(f"  Drop columns ({len(drop_cols)}): {drop_cols}")

    full_x = dataframe[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    reduced_x = dataframe[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)

    full_split = slice_splits(full_x, labels, split_indices)
    reduced_split = slice_splits(reduced_x, labels, split_indices)

    _, full_train, full_val, full_test = scale_split_data(
        full_split["train"][0],
        full_split["val"][0],
        full_split["test"][0],
    )
    _, reduced_train, reduced_val, reduced_test = scale_split_data(
        reduced_split["train"][0],
        reduced_split["val"][0],
        reduced_split["test"][0],
    )

    full_result = _train_projbridge(
        full_train,
        full_split["train"][1],
        full_val,
        full_split["val"][1],
        full_test,
        full_split["test"][1],
        full_x.shape[1],
        device,
        epochs,
        lr,
    )
    reduced_result = _train_projbridge(
        reduced_train,
        reduced_split["train"][1],
        reduced_val,
        reduced_split["val"][1],
        reduced_test,
        reduced_split["test"][1],
        reduced_x.shape[1],
        device,
        epochs,
        lr,
    )

    print(f"  Full features   : {full_result}")
    print(f"  Reduced features: {reduced_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit suspicious temporal fields in UAV/AP scenarios")
    parser.add_argument("--scenario", type=str, required=True, choices=["uav", "ap"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run_audit(args.scenario, epochs=args.epochs, lr=args.lr, device=args.device, seed=args.seed)
