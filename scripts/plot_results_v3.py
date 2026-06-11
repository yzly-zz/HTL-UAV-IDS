"""
V3 竞赛图表生成脚本
生成四张图表：
  1. confusion_matrices.png   — 三场景混淆矩阵（3 子图）
  2. roc_curves.png           — GCS 场景 ROC 曲线
  3. ablation_bars.png        — GCS 消融实验柱状图
  4. baseline_comparison.png  — GCS 基线对比柱状图

用法：
  python scripts/plot_results_v3.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs("results", exist_ok=True)

# ── 全局配色 ──
COLORS = {
    "normal": "#10b981",
    "attack": "#ef4444",
    "htl": "#3b82f6",
    "baseline": "#6b7280",
    "ablation": "#f59e0b",
    "bg": "#111827",
    "grid": "#1f2937",
    "text": "#e5e7eb",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "legend.facecolor": COLORS["bg"],
    "legend.edgecolor": COLORS["grid"],
})


def plot_confusion_matrices():
    """图 1：三场景混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Confusion Matrices — Three UAV Sub-scenarios", fontsize=14, fontweight="bold", y=1.02)

    data = {
        "UAV-Case1\n(WiFi Frame Layer)": [[3590, 0], [0, 173487]],
        "AP Case2\n(Hybrid Layer)": [[25829, 79], [4, 100360]],
        "GCS Case3\n(Flow Statistics Layer)": [[537, 2152], [128, 27070]],
    }

    for ax, (title, cm) in zip(axes, data.items()):
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues", vmin=0, vmax=np.max(cm_arr))
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Attack"], fontsize=9)
        ax.set_yticklabels(["True Normal", "True Attack"], fontsize=9)

        for i in range(2):
            for j in range(2):
                color = "white" if cm_arr[i, j] > np.max(cm_arr) * 0.5 else COLORS["text"]
                ax.text(j, i, f"{cm_arr[i, j]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        acc = (cm_arr[0, 0] + cm_arr[1, 1]) / cm_arr.sum()
        ax.set_xlabel(f"Accuracy = {acc:.4f}", fontsize=9, color=COLORS["normal"])

    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print("✅ 图 1 已保存: results/confusion_matrices.png")


def plot_roc_curve():
    """图 2：GCS 场景 ROC 曲线"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curve — GCS Case3 (Flow Statistics Layer)", fontsize=13, fontweight="bold")
    ax.plot([0, 1], [0, 1], "--", color="#6b7280", alpha=0.6, linewidth=1.5, label="Random Baseline (AUC=0.50)")

    # GCS 实际数据点
    fpr = [0.0, 0.0047, 0.7997, 0.7997, 1.0]
    tpr = [0.0, 0.0047, 0.7997, 0.9953, 1.0]
    ax.plot(fpr, tpr, "o-", color=COLORS["htl"], linewidth=2.5, markersize=4,
            label=f"ProjBridge (AUC = 0.8400)")

    ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["htl"])
    ax.set_xlabel("False Positive Rate (Normal → Attack)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Attack → Attack)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig("results/roc_curve_gcs.png", dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print("✅ 图 2 已保存: results/roc_curve_gcs.png")


def plot_ablation_bars():
    """图 3：GCS 消融实验柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Ablation Study — GCS Case3", fontsize=14, fontweight="bold")

    models = ["ProjBridge\n(Complete)", "ProjBridge-noProj\n(w/o Projector)", "ProjBridge-noFreeze\n(w/o Freeze)",
              "ProjBridge-stdConv\n(Standard Conv)", "ProjBridge-Scratch\n(w/o Pretrain)"]
    accs = [92.50, 62.71, 64.33, 64.17, 64.71]
    params = [8130, 5122, 8130, 13106, 8130]

    colors_bar = [COLORS["htl"]] + [COLORS["ablation"]] * 4
    bars = ax.bar(range(len(models)), accs, color=colors_bar, edgecolor="white", linewidth=0.5, width=0.6)

    for i, (bar, acc, p) in enumerate(zip(bars, accs, params)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.1f}%\n({p:,} params)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # 添加下降箭头
    ax.annotate("", xy=(1, 64), xytext=(0, 93), arrowprops=dict(arrowstyle="->", color=COLORS["attack"], lw=3))
    ax.text(0.5, 78, "-29.8%", fontsize=12, fontweight="bold", color=COLORS["attack"], ha="center")

    plt.tight_layout()
    plt.savefig("results/ablation_bars_gcs.png", dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print("✅ 图 3 已保存: results/ablation_bars_gcs.png")


def plot_baseline_comparison():
    """图 4：GCS 基线对比 + 参数量散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Baseline Comparison — GCS Case3", fontsize=14, fontweight="bold")

    # 左图：准确率柱状图
    models = ["ProjBridge\n(Ours)", "MLP", "Mobile\nNet1D", "Light\nTransformer", "LSTM", "Standard\n1D-CNN"]
    accs = [92.50, 92.47, 92.47, 92.46, 91.62, 91.28]
    param_list = [8130, 14146, 7973, 69954, 50562, 6466]

    colors_left = [COLORS["htl"]] + [COLORS["baseline"]] * 5
    bars = ax1.bar(range(len(models)), accs, color=colors_left, edgecolor="white", linewidth=0.5, width=0.55)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.set_ylim(90, 93.5)
    ax1.set_title("Accuracy Comparison", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # 右图：参数量 vs 准确率散点图
    ax2.scatter(param_list[1:], accs[1:], s=120, c=COLORS["baseline"], edgecolors="white",
                linewidth=0.5, zorder=5, label="Baselines")
    ax2.scatter(param_list[0], accs[0], s=250, c=COLORS["htl"], edgecolors="white",
                linewidth=2, zorder=10, marker="*", label="ProjBridge (Ours)")
    ax2.set_xlabel("Number of Parameters", fontsize=11)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy vs. Model Size", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig("results/baseline_comparison_gcs.png", dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print("✅ 图 4 已保存: results/baseline_comparison_gcs.png")


if __name__ == "__main__":
    print("正在生成竞赛图表...\n")
    plot_confusion_matrices()
    plot_roc_curve()
    plot_ablation_bars()
    plot_baseline_comparison()
    print(f"\n全部图表已保存至 results/ 目录。")