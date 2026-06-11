import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from config import (
    DEFAULT_SEED,
    EVAL_BATCH_SIZE,
    SCENARIOS,
    SHARED_DIM,
    SOURCE_MODEL_PATH,
    TARGET_BATCH_SIZE,
    WEIGHTS_DIR,
)
from src.models.hda_1dcnn import ProjBridge
from src.training.experiment_utils import (
    evaluate_classifier,
    load_pretrained_backbone,
    make_dataloader,
    prepare_target_experiment,
    run_training_loop,
    save_json,
    set_seed,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def _resolve_class_weights(scenario_key, y_train, class_weight_norm=None):
    if class_weight_norm is not None:
        return np.array([float(x) for x in class_weight_norm.split(",")], dtype=np.float32)

    default_weight = SCENARIOS[scenario_key].get("default_class_weight")
    if default_weight is not None:
        return np.array(default_weight, dtype=np.float32)

    return compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train).astype(np.float32)


def run_finetune(
    scenario_key,
    base_model_path=SOURCE_MODEL_PATH,
    epochs=20,
    class_weight_norm=None,
    lr=1e-4,
    device="cuda",
    seed=DEFAULT_SEED,
):
    set_seed(seed)
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key

    print(f"\n{'=' * 60}")
    print(f"  阶段二：目标域微调 -> {scenario['name']}")
    print(f"  设备: {device} | Epochs: {epochs} | LR: {lr} | Seed: {seed}")
    print(f"{'=' * 60}")

    split_path = WEIGHTS_DIR / f"target_split_{suffix}.npz"
    scaler_path = WEIGHTS_DIR / f"target_scaler_{suffix}.pkl"
    feature_path = WEIGHTS_DIR / f"target_feature_names_{suffix}.json"

    experiment = prepare_target_experiment(
        files=scenario["files"],
        label_col=scenario["label_col"],
        scaler_path=scaler_path,
        split_path=split_path,
        seed=seed,
        split_mode=scenario.get("split_mode", "random_stratified"),
        split_column=scenario.get("split_column"),
        block_size=scenario.get("block_size"),
    )

    with feature_path.open("w", encoding="utf-8") as handle:
        json.dump(experiment["feature_names"], handle, ensure_ascii=False, indent=2)

    X_train, y_train = experiment["train"]
    X_val, y_val = experiment["val"]
    X_test, y_test = experiment["test"]

    train_loader = make_dataloader(X_train, y_train, batch_size=TARGET_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    class_weights = _resolve_class_weights(scenario_key, y_train, class_weight_norm=class_weight_norm)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  类别权重: 正常={class_weights[0]:.3f}, 攻击={class_weights[1]:.3f}")
    print(f"  Split: train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")

    model = ProjBridge(input_dim=experiment["input_dim"], shared_dim=SHARED_DIM, num_classes=2).to(device)
    loaded_pretrained = load_pretrained_backbone(model, base_model_path, device=device)
    if loaded_pretrained:
        model.freeze_backbone_for_finetuning()
        print(f"  已加载预训练基座: {base_model_path}")
    else:
        print("  未找到预训练基座，将以 scratch 模式训练。")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc, history = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    val_metrics = evaluate_classifier(model, val_loader, device, criterion=criterion)
    test_metrics = evaluate_classifier(model, test_loader, device, criterion=criterion)

    model_path = WEIGHTS_DIR / f"target_uav_model_{suffix}.pth"
    torch.save(model.state_dict(), model_path)

    save_json(
        WEIGHTS_DIR / f"target_metrics_{suffix}.json",
        {
            "scenario": scenario_key,
            "input_dim": experiment["input_dim"],
            "class_weights": class_weights.tolist(),
            "best_val_acc": best_val_acc,
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "val_auc_roc": val_metrics["auc_roc"],
            "test_auc_roc": test_metrics["auc_roc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "history": history,
        },
    )

    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    print(f"  最终测试准确率: {test_metrics['accuracy']:.4f}")
    print(f"  模型已保存: {model_path}")

    return {
        "scenario": scenario_key,
        "input_dim": experiment["input_dim"],
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["accuracy"],
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "feature_names": experiment["feature_names"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProjBridge 目标域微调")
    parser.add_argument("--scenario", type=str, required=True, choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--class_weight", type=str, default=None)
    args = parser.parse_args()

    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            run_finetune(
                key,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                seed=args.seed,
                class_weight_norm=args.class_weight,
            )
    else:
        run_finetune(
            args.scenario,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            class_weight_norm=args.class_weight,
        )
