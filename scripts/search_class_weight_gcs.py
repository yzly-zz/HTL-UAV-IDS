import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, SCENARIOS, SHARED_DIM, SOURCE_MODEL_PATH, TARGET_BATCH_SIZE, WEIGHTS_DIR
from src.models.hda_1dcnn import ProjBridge
from src.training.experiment_utils import (
    evaluate_classifier,
    load_pretrained_backbone,
    make_dataloader,
    prepare_target_experiment,
    run_training_loop,
    set_seed,
)


def search(device_str, epochs=10, lr=1e-4, seed=DEFAULT_SEED):
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    scenario = SCENARIOS["gcs"]
    print(f"Device: {device}")

    experiment = prepare_target_experiment(
        files=scenario["files"],
        label_col=scenario["label_col"],
        scaler_path=WEIGHTS_DIR / "class_weight_search_scaler_gcs.pkl",
        split_path=WEIGHTS_DIR / "class_weight_search_split_gcs.npz",
        seed=seed,
    )

    X_train, y_train = experiment["train"]
    X_val, y_val = experiment["val"]
    train_loader = make_dataloader(X_train, y_train, batch_size=TARGET_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    w_norms = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    w_atks = [0.5, 0.6, 0.7, 0.8, 1.0]
    best = None

    print(f"Searching {len(w_norms) * len(w_atks)} weight combinations on train/val only...")
    for wn in w_norms:
        for wa in w_atks:
            model = ProjBridge(input_dim=experiment["input_dim"], shared_dim=SHARED_DIM, num_classes=2).to(device)
            if load_pretrained_backbone(model, SOURCE_MODEL_PATH, device):
                model.freeze_backbone_for_finetuning()

            criterion = nn.CrossEntropyLoss(weight=torch.tensor([wn, wa], dtype=torch.float32).to(device))
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            _, _ = run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=epochs,
            )
            metrics = evaluate_classifier(model, val_loader, device=device)
            labels = metrics["labels"]
            preds = metrics["preds"]
            f1m = f1_score(labels, preds, average="macro")
            f1n = f1_score(labels, preds, pos_label=0, zero_division=0)
            f1a = f1_score(labels, preds, pos_label=1, zero_division=0)

            if best is None or f1m > best["f1_macro"]:
                best = {
                    "normal_weight": wn,
                    "attack_weight": wa,
                    "f1_macro": f1m,
                    "f1_normal": f1n,
                    "f1_attack": f1a,
                }
                marker = " <- best"
            else:
                marker = ""
            print(
                f"  wn={wn:.2f} wa={wa:.2f} -> "
                f"F1-macro={f1m:.4f} (N={f1n:.4f}, A={f1a:.4f}){marker}"
            )

    print("=" * 60)
    print(
        f"Best weights: normal={best['normal_weight']:.3f}, "
        f"attack={best['attack_weight']:.3f}, "
        f"F1-macro={best['f1_macro']:.4f}"
    )
    print("Use them explicitly in finetune.py via:")
    print(f"  --class_weight {best['normal_weight']:.3f},{best['attack_weight']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    search(args.device, epochs=args.epochs, lr=args.lr, seed=args.seed)
