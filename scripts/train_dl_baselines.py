import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, SCENARIOS, TARGET_BATCH_SIZE, WEIGHTS_DIR
from src.models.baselines import BASELINE_MODELS
from src.training.experiment_utils import (
    evaluate_classifier,
    make_dataloader,
    prepare_target_experiment,
    run_training_loop,
    save_json,
    set_seed,
)


def train_one_model(model, train_loader, val_loader, test_loader, epochs, lr, device, class_weights_tensor):
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc, history = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )
    test_metrics = evaluate_classifier(model, test_loader, device, criterion=criterion)
    return best_val_acc, history, test_metrics


def run_baselines_for_scenario(scenario_key, epochs=20, lr=1e-4, device="cuda", seed=DEFAULT_SEED):
    set_seed(seed)
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key

    print(f"\n{'=' * 60}")
    print(f"  场景: {scenario['name']} -> 深度学习基线训练")
    print(f"  设备: {device} | Epochs: {epochs} | LR: {lr} | Seed: {seed}")
    print(f"{'=' * 60}")

    experiment = prepare_target_experiment(
        files=scenario["files"],
        label_col=scenario["label_col"],
        scaler_path=WEIGHTS_DIR / f"baseline_scaler_{suffix}.pkl",
        split_path=WEIGHTS_DIR / f"baseline_split_{suffix}.npz",
        seed=seed,
        split_mode=scenario.get("split_mode", "random_stratified"),
        split_column=scenario.get("split_column"),
        block_size=scenario.get("block_size"),
    )

    with (WEIGHTS_DIR / f"baseline_feature_names_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment["feature_names"], handle, ensure_ascii=False, indent=2)

    X_train, y_train = experiment["train"]
    X_val, y_val = experiment["val"]
    X_test, y_test = experiment["test"]

    train_loader = make_dataloader(X_train, y_train, batch_size=TARGET_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train).astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"  Split: train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
    print(f"  类别权重: 正常={class_weights[0]:.3f}, 攻击={class_weights[1]:.3f}")

    results = {}
    for name, model_cls in BASELINE_MODELS.items():
        model = model_cls(input_dim=experiment["input_dim"], num_classes=2).to(device)
        n_params = sum(param.numel() for param in model.parameters())
        best_val_acc, history, test_metrics = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            class_weights_tensor=class_weights_tensor,
        )

        torch.save(model.state_dict(), WEIGHTS_DIR / f"baseline_{name}_{suffix}.pth")
        save_json(
            WEIGHTS_DIR / f"baseline_metrics_{name}_{suffix}.json",
            {
                "best_val_acc": best_val_acc,
                "test_accuracy": test_metrics["accuracy"],
                "test_auc_roc": test_metrics["auc_roc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "history": history,
            },
        )
        results[name] = {
            "best_val_acc": best_val_acc,
            "test_acc": test_metrics["accuracy"],
            "params": n_params,
        }
        print(
            f"  {name}: best val acc={best_val_acc:.4f}, "
            f"test acc={test_metrics['accuracy']:.4f}, params={n_params:,}"
        )

    return results, experiment["input_dim"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 二分类深度学习基线训练")
    parser.add_argument("--scenario", type=str, default="all", choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    all_results = {}
    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            res, dim = run_baselines_for_scenario(key, epochs=args.epochs, lr=args.lr, device=args.device, seed=args.seed)
            all_results[key] = {"results": res, "input_dim": dim}
    else:
        res, dim = run_baselines_for_scenario(args.scenario, epochs=args.epochs, lr=args.lr, device=args.device, seed=args.seed)
        all_results[args.scenario] = {"results": res, "input_dim": dim}

    print(f"\n{'=' * 70}")
    print("  深度学习基线训练汇总")
    print(f"{'=' * 70}")
    for key, data in all_results.items():
        print(f"  {SCENARIOS[key]['name']} (dim={data['input_dim']})")
        for name, record in data["results"].items():
            print(
                f"    {name:<20s} val={record['best_val_acc']:.4f} "
                f"test={record['test_acc']:.4f} params={record['params']:,}"
            )
