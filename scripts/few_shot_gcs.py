import argparse
import copy
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, RESULTS_DIR, SCENARIOS, SHARED_DIM, SOURCE_MODEL_PATH, TARGET_BATCH_SIZE
from src.models.baselines import MobileNet1D, VanillaMLP
from src.models.hda_1dcnn import ProjBridge
from src.training.experiment_utils import (
    load_and_clean,
    load_pretrained_backbone,
    make_dataloader,
    set_seed,
)


def train_one_model(model, train_loader, val_loader, test_loader, epochs, lr, device, cw):
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_f1 = float("-inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        labels_all = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                outputs = model(inputs).argmax(dim=1).cpu().numpy()
                preds.extend(outputs)
                labels_all.extend(labels.numpy())
        val_f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs).argmax(dim=1).cpu().numpy()
            preds.extend(outputs)
            labels_all.extend(labels.numpy())

    preds = np.array(preds)
    labels_all = np.array(labels_all)
    return {
        "acc": float((preds == labels_all).mean()),
        "f1_macro": f1_score(labels_all, preds, average="macro", zero_division=0),
        "f1_normal": f1_score(labels_all, preds, pos_label=0, zero_division=0),
        "f1_attack": f1_score(labels_all, preds, pos_label=1, zero_division=0),
    }


def run_experiment(device="cuda", seeds=(42, 123, 456, 789, 1024)):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    set_seed(DEFAULT_SEED)
    scenario = SCENARIOS["gcs"]
    ratios = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    epochs, lr = 20, 1e-4

    X_raw, y, _ = load_and_clean(scenario["files"], label_col=scenario["label_col"])

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=DEFAULT_SEED,
        stratify=y,
    )
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=0.2,
        random_state=DEFAULT_SEED,
        stratify=y_train_pool,
    )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train_full)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    val_loader = make_dataloader(X_val_s, y_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test_s, y_test, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_full)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    pretrained = torch.load(SOURCE_MODEL_PATH, map_location=device) if os.path.exists(SOURCE_MODEL_PATH) else None

    all_results = defaultdict(list)
    for ratio in ratios:
        n_samples = max(2, int(len(y_train_full) * ratio))
        for seed in seeds:
            np.random.seed(seed)
            idx_normal = np.where(y_train_full == 0)[0]
            idx_attack = np.where(y_train_full == 1)[0]
            n_normal = max(1, int(n_samples * (1 - y_train_full.mean())))
            n_attack = max(1, n_samples - n_normal)

            sampled_normal = np.random.choice(idx_normal, min(n_normal, len(idx_normal)), replace=False)
            sampled_attack = np.random.choice(idx_attack, min(n_attack, len(idx_attack)), replace=False)
            sampled_idx = np.concatenate([sampled_normal, sampled_attack])
            np.random.shuffle(sampled_idx)

            X_train = scaler.transform(X_train_full[sampled_idx])
            y_train = y_train_full[sampled_idx]
            train_loader = make_dataloader(X_train, y_train, batch_size=min(TARGET_BATCH_SIZE, len(y_train)), shuffle=True)

            projbridge = ProjBridge(input_dim=X_raw.shape[1], shared_dim=SHARED_DIM, num_classes=2).to(device)
            if pretrained is not None:
                projbridge.load_state_dict(
                    {
                        **projbridge.state_dict(),
                        **{
                            key: value
                            for key, value in pretrained.items()
                            if key in projbridge.state_dict() and "projector" not in key and "classifier" not in key
                        },
                    }
                )
                projbridge.freeze_backbone_for_finetuning()
            result = train_one_model(projbridge, train_loader, val_loader, test_loader, epochs, lr, device, cw_t)
            result.update(ratio=ratio, n_samples=n_samples, seed=seed, model="ProjBridge")
            all_results["ProjBridge"].append(result)

            mlp = VanillaMLP(input_dim=X_raw.shape[1], num_classes=2).to(device)
            result = train_one_model(mlp, train_loader, val_loader, test_loader, epochs, lr, device, cw_t)
            result.update(ratio=ratio, n_samples=n_samples, seed=seed, model="MLP")
            all_results["MLP"].append(result)

            mobilen = MobileNet1D(input_dim=X_raw.shape[1], num_classes=2).to(device)
            result = train_one_model(mobilen, train_loader, val_loader, test_loader, epochs, lr, device, cw_t)
            result.update(ratio=ratio, n_samples=n_samples, seed=seed, model="MobileNet1D")
            all_results["MobileNet1D"].append(result)

    summary_rows = []
    for model_name in ["ProjBridge", "MLP", "MobileNet1D"]:
        for ratio in ratios:
            group = [record for record in all_results[model_name] if record["ratio"] == ratio]
            if not group:
                continue
            summary_rows.append(
                {
                    "model": model_name,
                    "ratio": ratio,
                    "n_samples": group[0]["n_samples"],
                    "acc_mean": np.mean([item["acc"] for item in group]),
                    "acc_std": np.std([item["acc"] for item in group]),
                    "f1_macro_mean": np.mean([item["f1_macro"] for item in group]),
                    "f1_macro_std": np.std([item["f1_macro"] for item in group]),
                    "f1_normal_mean": np.mean([item["f1_normal"] for item in group]),
                    "f1_normal_std": np.std([item["f1_normal"] for item in group]),
                    "f1_attack_mean": np.mean([item["f1_attack"] for item in group]),
                    "f1_attack_std": np.std([item["f1_attack"] for item in group]),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "few_shot_gcs.csv", index=False, encoding="utf-8")
    print(f"Saved: {RESULTS_DIR / 'few_shot_gcs.csv'}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Transfer Learning Experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()
    run_experiment(device=args.device, seeds=args.seeds)
