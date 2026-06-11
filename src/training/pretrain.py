import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from config import DEFAULT_SEED, SHARED_DIM, SOURCE_BATCH_SIZE, SOURCE_MODEL_PATH, SOURCE_VAL_RATIO
from src.models.hda_1dcnn import ProjBridge
from src.training.experiment_utils import (
    load_and_clean,
    make_dataloader,
    run_training_loop,
    save_split_indices,
    save_json,
    scale_split_data,
    set_seed,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def run_pretrain(source_data_path, batch_size=SOURCE_BATCH_SIZE, epochs=20, lr=1e-3, device="cuda", seed=DEFAULT_SEED):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    source_path = Path(source_data_path)

    print(f"========== [阶段一] 开始源域预训练 | 设备: {device} ==========")

    X_raw, y, _ = load_and_clean([source_path], label_col=None)
    input_dim = X_raw.shape[1]

    indices = torch.arange(len(y)).numpy()
    stratify = y if len(set(y.tolist())) > 1 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=SOURCE_VAL_RATIO,
        random_state=seed,
        stratify=stratify,
    )

    X_train = X_raw[train_idx]
    y_train = y[train_idx]
    X_val = X_raw[val_idx]
    y_val = y[val_idx]

    scaler, X_train_scaled, X_val_scaled, _ = scale_split_data(X_train, X_val, X_val)

    train_loader = make_dataloader(X_train_scaled, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val_scaled, y_val, batch_size=batch_size, shuffle=False)

    model = ProjBridge(input_dim=input_dim, shared_dim=SHARED_DIM, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
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

    SOURCE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SOURCE_MODEL_PATH)
    save_split_indices(SOURCE_MODEL_PATH.parent / "source_split_indices.npz", (train_idx, val_idx, val_idx))
    save_json(
        SOURCE_MODEL_PATH.parent / "source_pretrain_metrics.json",
        {
            "input_dim": input_dim,
            "best_val_acc": best_val_acc,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "history": history,
        },
    )
    print(f"[阶段一完成] 最佳验证准确率: {best_val_acc:.4f}")
    print(f"[阶段一完成] 源域基座模型已保存至: {SOURCE_MODEL_PATH}")


if __name__ == "__main__":
    run_pretrain("data/raw/CIC-ToN-IoT-V2.parquet")
