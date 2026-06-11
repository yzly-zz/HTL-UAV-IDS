# ProjBridge

ProjBridge is a lightweight transfer-learning framework for UAV intrusion detection.
The current codebase targets three UAV-NIDD sub-scenarios and a ToN-IoT source-domain pretraining stage.

## Status

This repository is under active research refactoring.
The model name is `ProjBridge`.
The old project name `HTL-UAV-IDS` is deprecated and should be treated as a legacy alias only.

## What Changed

- Unified experiment configuration now lives in `config.py`.
- Training now follows a strict `train / val / test` protocol.
- Best checkpoints are selected on `val`, not on `test`.
- Scenario-specific splits are persisted and reused by evaluation scripts.
- `UAV-Case1` and `AP-Case2` now use block-aware splitting based on `frame.number` to reduce temporal-neighbor leakage.

## Repository Layout

```text
ProjBridge/
├── config.py
├── requirements.txt
├── data/
│   └── raw/
├── results/
├── scripts/
├── src/
│   ├── data_engine/
│   ├── inference/
│   ├── models/
│   ├── training/
│   └── web_gcs/
└── weights/
```

## Core Files

- `src/models/hda_1dcnn.py`: ProjBridge model definition.
- `src/training/pretrain.py`: source-domain pretraining.
- `src/training/finetune.py`: target-domain finetuning.
- `src/training/experiment_utils.py`: shared split, scaler, evaluation, and checkpoint utilities.
- `scripts/evaluate_v3.py`: evaluation on persisted splits.
- `scripts/train_dl_baselines.py`: deep-learning baselines.
- `scripts/train_ablations_v3.py`: ablation study.
- `scripts/few_shot_gcs.py`: few-shot experiments for GCS.

## Run

```bash
pip install -r requirements.txt
python scripts/pretrain.py
python scripts/finetune.py --scenario all
python scripts/train_dl_baselines.py --scenario all
python scripts/train_ablations_v3.py --scenario all
python scripts/evaluate_v3.py --device cuda
```

## Research Notes

- `UAV-Case1` and `AP-Case2` remain scientifically high-risk because perfect or near-perfect accuracy persists even after protocol cleanup.
- `GCS-Case3` is currently the only scenario that behaves like a credible main benchmark.
- If this project is used for a paper, the contribution should center on `GCS-Case3`, transfer learning, ablation evidence, and robustness analysis, not on the perfect-score scenarios.
