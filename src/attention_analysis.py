"""
Attention Dynamics and Weight Distribution Analyzer for Attention Fusion Models.
Computes and visualizes how multi-modal / multi-backbone branches are weighted per DR grade.
"""

import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DR_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def summarize_attention_weights(
    attention_weights: np.ndarray,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    branch_names: Tuple[str, str] = ("ResNet-18", "EfficientNet-B0")
) -> pd.DataFrame:
    """
    Creates a detailed DataFrame summarizing learned branch attention weights per DR severity grade.

    Args:
        attention_weights (np.ndarray): Array of shape (N, 2) containing softmax attention weights.
        y_true (np.ndarray): Ground truth labels (N,).
        y_pred (np.ndarray, optional): Predicted labels (N,).
        branch_names (tuple): Names of the two branches.

    Returns:
        pd.DataFrame: Summary table with mean, std, median per class.
    """
    records = []
    w0 = attention_weights[:, 0]
    w1 = attention_weights[:, 1]

    # Global summary
    records.append({
        "DR Grade": "Overall (All Classes)",
        f"{branch_names[0]} Mean (Std)": f"{np.mean(w0):.4f} ± {np.std(w0):.4f}",
        f"{branch_names[1]} Mean (Std)": f"{np.mean(w1):.4f} ± {np.std(w1):.4f}",
        f"{branch_names[0]} Median [IQR]": f"{np.median(w0):.4f} [{np.percentile(w0, 75) - np.percentile(w0, 25):.4f}]",
        f"{branch_names[1]} Median [IQR]": f"{np.median(w1):.4f} [{np.percentile(w1, 75) - np.percentile(w1, 25):.4f}]",
        "Sample Count": len(y_true)
    })

    # Per-class summary
    for c in range(len(DR_CLASSES)):
        mask = (y_true == c)
        if np.sum(mask) == 0:
            continue
        c_w0 = w0[mask]
        c_w1 = w1[mask]
        records.append({
            "DR Grade": f"Grade {c} ({DR_CLASSES[c]})",
            f"{branch_names[0]} Mean (Std)": f"{np.mean(c_w0):.4f} ± {np.std(c_w0):.4f}",
            f"{branch_names[1]} Mean (Std)": f"{np.mean(c_w1):.4f} ± {np.std(c_w1):.4f}",
            f"{branch_names[0]} Median [IQR]": f"{np.median(c_w0):.4f} [{np.percentile(c_w0, 75) - np.percentile(c_w0, 25):.4f}]",
            f"{branch_names[1]} Median [IQR]": f"{np.median(c_w1):.4f} [{np.percentile(c_w1, 75) - np.percentile(c_w1, 25):.4f}]",
            "Sample Count": int(np.sum(mask))
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def plot_attention_distribution_by_grade(
    attention_weights: np.ndarray,
    y_true: np.ndarray,
    branch_names: Tuple[str, str] = ("ResNet-18", "EfficientNet-B0"),
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots boxplots and bar plots of attention weights across all 5 DR severity grades.
    """
    df = pd.DataFrame({
        "DR Grade": [DR_CLASSES[int(label)] for label in y_true],
        "Grade Index": y_true,
        branch_names[0]: attention_weights[:, 0],
        branch_names[1]: attention_weights[:, 1]
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Stacked / Grouped Mean Attention per Grade
    means = df.groupby("Grade Index")[[branch_names[0], branch_names[1]]].mean()
    means.index = [DR_CLASSES[i] for i in means.index]

    means.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="black",
        alpha=0.85
    )
    axes[0].set_title("Mean Branch Contribution per DR Grade (Normalized)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("DR Severity Grade", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Average Attention Weight", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(title="Backbone Branch", loc="upper right")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Boxplot distribution of ResNet branch weight by Grade
    sns.boxplot(
        data=df,
        x="DR Grade",
        y=branch_names[0],
        ax=axes[1],
        palette="Blues",
        order=DR_CLASSES
    )
    axes[1].set_title(f"Distribution of {branch_names[0]} Weight across DR Grades", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("DR Severity Grade", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Attention Weight", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
