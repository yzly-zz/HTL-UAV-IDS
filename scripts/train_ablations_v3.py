import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, SCENARIOS, SHARED_DIM, SOURCE_MODEL_PATH, TARGET_BATCH_SIZE, WEIGHTS_DIR
from src.models.hda_1dcnn import DepthwiseSeparableConv1d, DomainProjector, ProjBridge
from src.training.experiment_utils import (
    evaluate_classifier,
    load_pretrained_backbone,
    make_dataloader,
    prepare_target_experiment,
    run_training_loop,
    save_json,
    set_seed,
)


class ProjBridgeNoProj(nn.Module):
    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super().__init__()
        self.shared_dim = shared_dim
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 32, 3, 1),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(32, 64, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        if x.shape[1] < self.shared_dim:
            padding = torch.zeros(x.shape[0], self.shared_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.shape[1] > self.shared_dim:
            x = x[:, : self.shared_dim]
        return self.classifier(self.backbone(x.unsqueeze(1)).squeeze(-1))


class ProjBridgeStdConv(nn.Module):
    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super().__init__()
        self.projector = DomainProjector(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(self.projector(x).unsqueeze(1)).squeeze(-1))


ABLATION_VARIANTS = {
    "ProjBridge-noProj": {
        "cls": ProjBridgeNoProj,
        "use_pretrained": True,
        "freeze_backbone": True,
    },
    "ProjBridge-noFreeze": {
        "cls": ProjBridge,
        "use_pretrained": True,
        "freeze_backbone": False,
    },
    "ProjBridge-stdConv": {
        "cls": ProjBridgeStdConv,
        "use_pretrained": True,
        "freeze_backbone": True,
    },
    "ProjBridge-Scratch": {
        "cls": ProjBridge,
        "use_pretrained": False,
        "freeze_backbone": True,
    },
}


def _freeze_backbone_if_needed(model):
    if hasattr(model, "freeze_backbone_for_finetuning"):
        model.freeze_backbone_for_finetuning()
        return
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False


def run_ablations_for_scenario(scenario_key, epochs=20, lr=1e-4, device="cuda", seed=DEFAULT_SEED):
    set_seed(seed)
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key

    print(f"\n{'=' * 60}")
    print(f"  消融实验: {scenario['name']}")
    print(f"  设备: {device} | Epochs: {epochs} | LR: {lr} | Seed: {seed}")
    print(f"{'=' * 60}")

    experiment = prepare_target_experiment(
        files=scenario["files"],
        label_col=scenario["label_col"],
        scaler_path=WEIGHTS_DIR / f"ablation_scaler_{suffix}.pkl",
        split_path=WEIGHTS_DIR / f"ablation_split_{suffix}.npz",
        seed=seed,
        split_mode=scenario.get("split_mode", "random_stratified"),
        split_column=scenario.get("split_column"),
        block_size=scenario.get("block_size"),
    )

    X_train, y_train = experiment["train"]
    X_val, y_val = experiment["val"]
    X_test, y_test = experiment["test"]

    train_loader = make_dataloader(X_train, y_train, batch_size=TARGET_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train).astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    results = {}
    for variant_key, variant_info in ABLATION_VARIANTS.items():
        model = variant_info["cls"](input_dim=experiment["input_dim"], shared_dim=SHARED_DIM, num_classes=2).to(device)
        if variant_info["use_pretrained"]:
            load_pretrained_backbone(model, SOURCE_MODEL_PATH, device=device)
        if variant_info["freeze_backbone"]:
            _freeze_backbone_if_needed(model)

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
        test_metrics = evaluate_classifier(model, test_loader, device=device, criterion=criterion)
        params = sum(param.numel() for param in model.parameters())

        torch.save(model.state_dict(), WEIGHTS_DIR / f"ablation_{variant_key}_{suffix}.pth")
        save_json(
            WEIGHTS_DIR / f"ablation_metrics_{variant_key}_{suffix}.json",
            {
                "best_val_acc": best_val_acc,
                "test_accuracy": test_metrics["accuracy"],
                "test_auc_roc": test_metrics["auc_roc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "params": params,
                "history": history,
            },
        )
        results[variant_key] = {
            "best_val_acc": best_val_acc,
            "test_acc": test_metrics["accuracy"],
            "params": params,
        }
        print(
            f"  {variant_key}: "
            f"best val acc={best_val_acc:.4f}, "
            f"test acc={test_metrics['accuracy']:.4f}, "
            f"params={params:,}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 消融实验")
    parser.add_argument("--scenario", type=str, default="gcs", choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    all_results = {}
    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            all_results[key] = run_ablations_for_scenario(key, epochs=args.epochs, lr=args.lr, device=args.device, seed=args.seed)
    else:
        all_results[args.scenario] = run_ablations_for_scenario(
            args.scenario,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )

    print(f"\n{'=' * 70}")
    print("  消融实验汇总")
    print(f"{'=' * 70}")
    for key, results in all_results.items():
        print(f"  {SCENARIOS[key]['name']}")
        for variant_key, record in results.items():
            print(
                f"    {variant_key:<15s} val={record['best_val_acc']:.4f} "
                f"test={record['test_acc']:.4f} params={record['params']:,}"
            )
