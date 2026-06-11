#!/bin/bash
# 赋予执行权限: chmod +x scripts/run_experiment.sh

echo "==================================================="
echo "  [ProjBridge] 自动化实验流水线启动"
echo "==================================================="

echo "[1/2] 开始执行源域 (ToN-IoT) 预训练..."
python src/training/pretrain.py

echo "[2/2] 开始执行目标域 (UAV-NIDD) 异构微调..."
python src/training/finetune.py

echo "==================================================="
echo "  全部模型训练已完成！权重已保存至 weights/ 目录。"
echo "==================================================="