# HTL-UAV-IDS: Heterogeneous Transfer Learning for UAV Intrusion Detection

这是一个面向资源受限无人机 (UAV) 边缘计算环境的轻量级入侵检测系统。本项目创新性地提出了一种基于异构迁移学习 (HTL) 的 1D-CNN 架构，能够在无需特征维度对齐的前提下，将通用物联网流量中的时序先验知识迁移至无人机特定网络环境中。

# ============================================================
# HTL-UAV-IDS-V2 实验流水线（共 10 步）
# ============================================================

# 1. 预训练（源域）
python scripts/pretrain.py
# 2. 目标域微调（原版 HTL-UAV-IDS，作为强对比基线）
python scripts/finetune.py
# 3. 训练深度学习基线（MLP, CNN, LSTM, MobileNet1D, LightTransformer）
python scripts/train_dl_baselines.py
# 4. 训练传统机器学习基线（RF, XGBoost）
python scripts/train_baselines.py
# 5. 消融变体训练（HTL-noProj, HTL-noFreeze, HTL-stdConv, HTL-Scratch）
python scripts/train_ablations.py
# 6. 【关键】增强版微调（HTL-UAV-IDS-V2）
python scripts/finetune_enhanced.py
# 7. 综合评估与结果生成
python scripts/evaluate_all.py
# 8. 绘制图表（混淆矩阵、ROC、消融柱状图）
python scripts/plot_results.py
# 9. 样本效率实验
python scripts/sample_efficiency.py
# 10. 对抗鲁棒性评估（所有模型对比）
python scripts/evaluate_adversarial_all.py
# 11. 边缘推理基准测试
python scripts/benchmark_edge.py
# 12.启动地面站与边缘推理演示（需两个终端）（可选）
# 终端 1 —— 启动 GCS（地面控制站）后端
python src/web_gcs/app.py
# 终端 2 —— 启动边缘推理引擎（实时检测数据流）
python src/inference/real_time.py
# 浏览器访问 http://127.0.0.1:8000 查看态势感知大屏

## 🌟 核心创新点
* **异构维度对齐机制**：通过动态域投影层(Domain Projector)，解决源域与目标域特征维度不匹配问题。
* **极致轻量化架构**：引入深度可分离卷积(Depthwise Separable Convolution)，参数量仅10.8k，适用于资源极度受限的无人机边缘设备。
* **SHAP-Lite 边缘审计**：基于梯度乘积的微秒级特征归因算法，提供可信的安全审计日志。
* **全异步系统架构**：解耦边缘端高速抓包与模型推理，配合 WebSocket 实现毫秒级地面站态势感知广播。

## 📁 项目目录结构
```text
    ├── data/                   # 数据集目录 (包含 raw/ 和 processed/)
    ├── logs/                   # 运行日志与拦截报文存储
    ├── results/                # 实验评估图表与指标 (如混淆矩阵、ROC等)
    ├── scripts/                # 自动化运维与实验脚本
    ├── src/                    # 核心源代码
    │   ├── data_engine/        # 异构数据清洗与管道
    │   ├── inference/          # 边缘实时推理与 SHAP-Lite 解释器
    │   ├── models/             # 网络架构设计 (HTL-1DCNN 及基线模型)
    │   ├── training/           # 预训练与微调逻辑
    │   └── web_gcs/            # 地面站 FastAPI 后端与前端大屏
    ├── weights/                # 预训练基座与微调模型权重、Scaler
    ├── requirements.txt        # 依赖清单
    └── README.md               # 项目说明文档

# 通过执行提供的 Shell 脚本，一键完成源域预训练与目标域微调
