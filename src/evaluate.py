import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

def evaluate_predictions(y_true, y_pred, y_prob=None, model_name="Model"):
    """Tính các thước đo Accuracy, Precision, Recall, F1, AUC."""
    res = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "F1-Score": float(f1_score(y_true, y_pred)),
    }
    if y_prob is not None:
        res["AUC (ROC)"] = float(roc_auc_score(y_true, y_prob))
    return res

def plot_comparison_grid(df_results, title="So sánh các mô hình", save_path=None):
    """Vẽ ma trận 2x3 biểu đồ cột so sánh các mô hình."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC (ROC)"]
    order = df_results["Model"].tolist()
    
    colors = ["#CC3059", "#4CA655", "#566ECA", "#E0873F", "#8A2F9E", "#58C6E0", "#2A9D8F", "#F2C94C"]
    palette = {m: colors[i % len(colors)] for i, m in enumerate(order)}
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        if metric not in df_results.columns:
            continue
        sns.barplot(
            data=df_results, x="Model", y=metric, order=order,
            palette=palette, hue="Model", legend=False, ax=ax
        )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        lo = max(0.5, df_results[metric].min() - 0.03)
        ax.set_ylim(lo, min(1.0, df_results[metric].max() + 0.025))
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_roc_curves(curves, title="Đường ROC", save_path=None):
    """Vẽ đường cong ROC so sánh nhiều mô hình."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_true, y_prob in curves:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ngẫu nhiên")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
