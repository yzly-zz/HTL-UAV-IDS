import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_SEED, SCENARIOS
from src.training.experiment_utils import (
    load_dataframe_and_clean,
    make_block_split_indices,
    make_split_indices,
    save_json,
    scale_split_data,
    slice_splits,
)


ML_BASELINES = {
    "LogReg": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=DEFAULT_SEED,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=DEFAULT_SEED,
        n_jobs=-1,
    ),
    "HistGBDT": HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=DEFAULT_SEED,
    ),
}


def _get_split_indices(dataframe, labels, scenario, seed):
    if scenario.get("split_mode") == "block_stratified":
        return make_block_split_indices(
            dataframe=dataframe,
            labels=labels,
            split_column=scenario["split_column"],
            block_size=scenario["block_size"],
            seed=seed,
        )
    return make_split_indices(labels, seed=seed)


def _maybe_subsample(x, y, max_samples, seed):
    if max_samples is None or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return x[idx], y[idx]


def run_ml_baselines(scenario_key, max_train_samples=None, max_test_samples=None, seed=DEFAULT_SEED):
    scenario = SCENARIOS[scenario_key]
    print(f"\n{'=' * 60}")
    print(f"  ML baselines: {scenario['name']}")
    print(f"  Seed: {seed}")
    print(f"{'=' * 60}")

    dataframe, features, labels, feature_names = load_dataframe_and_clean(
        scenario["files"],
        label_col=scenario["label_col"],
    )
    split_indices = _get_split_indices(dataframe, labels, scenario, seed)
    split_data = slice_splits(features, labels, split_indices)
    scaler, x_train, x_val, x_test = scale_split_data(
        split_data["train"][0],
        split_data["val"][0],
        split_data["test"][0],
    )

    y_train = split_data["train"][1]
    y_val = split_data["val"][1]
    y_test = split_data["test"][1]

    x_train, y_train = _maybe_subsample(x_train, y_train, max_train_samples, seed)
    x_test, y_test = _maybe_subsample(x_test, y_test, max_test_samples, seed)

    print(f"  Features: {len(feature_names)}")
    print(f"  Split: train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")

    results = {}
    for name, model in ML_BASELINES.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            attack_scores = model.predict_proba(x_test)[:, 1]
        elif hasattr(model, "decision_function"):
            attack_scores = model.decision_function(x_test)
        else:
            attack_scores = None

        record = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_normal": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
            "f1_attack": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["Normal", "Attack"],
                output_dict=True,
                zero_division=0,
            ),
        }
        if attack_scores is not None and len(np.unique(y_test)) > 1:
            record["auc_roc"] = roc_auc_score(y_test, attack_scores)
        else:
            record["auc_roc"] = None
        results[name] = record
        print(
            f"  {name:<14s} "
            f"acc={record['accuracy']:.4f} "
            f"f1_macro={record['f1_macro']:.4f} "
            f"f1_normal={record['f1_normal']:.4f} "
            f"f1_attack={record['f1_attack']:.4f}"
        )

    save_json(
        os.path.join("results", f"ml_baselines_{scenario_key}.json"),
        {
            "scenario": scenario_key,
            "seed": seed,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "results": results,
        },
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train traditional ML baselines on ProjBridge splits")
    parser.add_argument("--scenario", type=str, default="gcs", choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            run_ml_baselines(
                key,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
            )
    else:
        run_ml_baselines(
            args.scenario,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )
