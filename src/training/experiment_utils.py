import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import DEFAULT_SEED, EVAL_BATCH_SIZE, RESULTS_DIR, TARGET_TEST_RATIO, TARGET_VAL_RATIO, WEIGHTS_DIR
from src.data_engine.dataset import IDSStreamDataset
from src.data_engine.preprocessor import CSV_ENCODINGS, HeterogeneousDataPreprocessor


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_read_csv(file_path):
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码: {file_path}")


def load_table(file_path):
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        return safe_read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"不支持的文件格式: {path}")


def load_and_clean(files, label_col=None):
    preprocessor = HeterogeneousDataPreprocessor()
    tables = [load_table(path) for path in files]
    full_df = pd.concat(tables, axis=0, ignore_index=True)
    features, labels, feature_names = preprocessor._clean_dataframe(full_df, label_col=label_col)
    return features, labels, feature_names


def load_dataframe_and_clean(files, label_col=None):
    preprocessor = HeterogeneousDataPreprocessor()
    tables = [load_table(path) for path in files]
    full_df = pd.concat(tables, axis=0, ignore_index=True)
    features, labels, feature_names = preprocessor._clean_dataframe(full_df, label_col=label_col)
    return full_df, features, labels, feature_names


def _stratify_target(labels):
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        return None
    return labels


def make_split_indices(labels, test_ratio=TARGET_TEST_RATIO, val_ratio=TARGET_VAL_RATIO, seed=DEFAULT_SEED):
    indices = np.arange(len(labels))
    stratify = _stratify_target(labels)

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )

    train_val_labels = labels[train_val_idx]
    val_ratio_within_train_val = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        random_state=seed,
        stratify=_stratify_target(train_val_labels),
    )
    return train_idx, val_idx, test_idx


def _block_ids_from_sorted_index(sorted_index, block_size):
    block_ids = np.empty(len(sorted_index), dtype=np.int64)
    for block_id, start in enumerate(range(0, len(sorted_index), block_size)):
        stop = min(start + block_size, len(sorted_index))
        block_ids[sorted_index[start:stop]] = block_id
    return block_ids


def make_block_split_indices(
    dataframe,
    labels,
    split_column,
    block_size,
    test_ratio=TARGET_TEST_RATIO,
    val_ratio=TARGET_VAL_RATIO,
    seed=DEFAULT_SEED,
):
    if split_column not in dataframe.columns:
        raise ValueError(f"block split column not found: {split_column}")

    order = np.argsort(pd.to_numeric(dataframe[split_column], errors="coerce").fillna(0).to_numpy(), kind="mergesort")
    block_ids = _block_ids_from_sorted_index(order, block_size)
    unique_blocks = np.unique(block_ids)

    block_labels = []
    for block_id in unique_blocks:
        block_y = labels[block_ids == block_id]
        values, counts = np.unique(block_y, return_counts=True)
        block_labels.append(values[np.argmax(counts)])
    block_labels = np.array(block_labels)

    stratify_blocks = block_labels if _stratify_target(block_labels) is not None else None
    train_val_blocks, test_blocks = train_test_split(
        unique_blocks,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_blocks,
    )

    train_val_block_labels = block_labels[np.isin(unique_blocks, train_val_blocks)]
    val_ratio_within_train_val = val_ratio / (1.0 - test_ratio)
    stratify_train_val_blocks = train_val_block_labels if _stratify_target(train_val_block_labels) is not None else None
    train_blocks, val_blocks = train_test_split(
        train_val_blocks,
        test_size=val_ratio_within_train_val,
        random_state=seed,
        stratify=stratify_train_val_blocks,
    )

    train_idx = np.where(np.isin(block_ids, train_blocks))[0]
    val_idx = np.where(np.isin(block_ids, val_blocks))[0]
    test_idx = np.where(np.isin(block_ids, test_blocks))[0]
    return train_idx, val_idx, test_idx


def slice_splits(features, labels, split_indices):
    train_idx, val_idx, test_idx = split_indices
    return {
        "train": (features[train_idx], labels[train_idx]),
        "val": (features[val_idx], labels[val_idx]),
        "test": (features[test_idx], labels[test_idx]),
    }


def scale_split_data(train_x, val_x, test_x, scaler=None):
    scaler = scaler or StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    val_scaled = scaler.transform(val_x)
    test_scaled = scaler.transform(test_x)
    return scaler, train_scaled, val_scaled, test_scaled


def make_dataloader(features, labels, batch_size, shuffle):
    dataset = IDSStreamDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_split_indices(path, split_indices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_idx=split_indices[0],
        val_idx=split_indices[1],
        test_idx=split_indices[2],
    )


def load_split_indices(path):
    data = np.load(path)
    return data["train_idx"], data["val_idx"], data["test_idx"]


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_pretrained_backbone(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return False

    pretrained_dict = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    filtered = {
        key: value
        for key, value in pretrained_dict.items()
        if key in model_dict and "projector" not in key and "classifier" not in key
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    return True


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_metric = float("-inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        val_metrics = evaluate_classifier(model, val_loader, device, criterion=criterion)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(epoch_record)

        if val_metrics["accuracy"] > best_metric:
            best_metric = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metric, history


def evaluate_classifier(model, data_loader, device, criterion=None):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            if criterion is not None:
                total_loss += criterion(logits, labels).item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)
    attack_probs = probs_np[:, 1] if len(probs_np) else np.array([])

    metrics = {
        "loss": total_loss / max(len(data_loader), 1) if criterion is not None else None,
        "accuracy": accuracy_score(labels_np, preds_np) if len(labels_np) else 0.0,
        "confusion_matrix": confusion_matrix(labels_np, preds_np).tolist() if len(labels_np) else [],
        "classification_report": classification_report(
            labels_np,
            preds_np,
            target_names=["Normal", "Attack"],
            digits=4,
            output_dict=True,
            zero_division=0,
        ) if len(labels_np) else {},
        "labels": labels_np,
        "preds": preds_np,
        "attack_probs": attack_probs,
    }

    if len(np.unique(labels_np)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels_np, attack_probs)
        metrics["pr_auc"] = average_precision_score(labels_np, attack_probs)
    else:
        metrics["auc_roc"] = None
        metrics["pr_auc"] = None
    return metrics


def prepare_target_experiment(
    files,
    label_col=None,
    scaler_path=None,
    split_path=None,
    seed=DEFAULT_SEED,
    split_mode="random_stratified",
    split_column=None,
    block_size=None,
):
    dataframe, features, labels, feature_names = load_dataframe_and_clean(files, label_col=label_col)
    if split_mode == "block_stratified":
        split_indices = make_block_split_indices(
            dataframe=dataframe,
            labels=labels,
            split_column=split_column,
            block_size=block_size,
            seed=seed,
        )
    else:
        split_indices = make_split_indices(labels, seed=seed)
    if split_path is not None:
        save_split_indices(split_path, split_indices)

    split_data = slice_splits(features, labels, split_indices)
    scaler, train_x, val_x, test_x = scale_split_data(
        split_data["train"][0],
        split_data["val"][0],
        split_data["test"][0],
    )
    if scaler_path is not None:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return {
        "feature_names": feature_names,
        "input_dim": features.shape[1],
        "split_indices": split_indices,
        "train": (train_x, split_data["train"][1]),
        "val": (val_x, split_data["val"][1]),
        "test": (test_x, split_data["test"][1]),
        "scaler": scaler,
    }
