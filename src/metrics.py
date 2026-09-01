"""
Comprehensive 11-Metric Clinical Evaluation Suite for Diabetic Retinopathy Screening.
Calculates Accuracy, Precision, Recall, Macro F1, Weighted F1, Balanced Accuracy,
Quadratic Weighted Kappa (QWK), Cohen's Kappa, MCC, Macro ROC-AUC, Log Loss,
Expected Calibration Error (ECE), and generates publication-grade evaluation curves.
"""

import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    average_precision_score
)

DR_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and returns reliability diagram points.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Computes the full suite of 11+ clinical and statistical metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cohen_k = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # ROC-AUC and Log Loss
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    try:
        ll = log_loss(y_true, y_prob)
    except Exception:
        ll = float("nan")

    # ECE and mAP
    ece, _, _, _ = compute_expected_calibration_error(y_true, y_prob)

    try:
        # Binarize labels for macro Average Precision
        num_classes = y_prob.shape[1]
        y_one_hot = np.eye(num_classes)[y_true]
        macro_ap = average_precision_score(y_one_hot, y_prob, average="macro")
    except Exception:
        macro_ap = float("nan")

    return {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Macro Precision": round(float(prec_macro), 4),
        "Macro Recall": round(float(rec_macro), 4),
        "Macro F1": round(float(f1_macro), 4),
        "Weighted F1": round(float(f1_weighted), 4),
        "Balanced Accuracy": round(float(balanced_acc), 4),
        "QWK": round(float(qwk), 4),
        "Cohen Kappa": round(float(cohen_k), 4),
        "MCC": round(float(mcc), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "Log Loss": round(float(ll), 4),
        "ECE": round(float(ece), 4),
        "Macro AP": round(float(macro_ap), 4)
    }


def compute_per_class_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Generates a per-class DataFrame with Precision, Recall, F1-Score, and Support.
    """
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(DR_CLASSES))), zero_division=0
    )
    df = pd.DataFrame({
        "Grade": list(range(len(DR_CLASSES))),
        "DR Severity": DR_CLASSES,
        "Precision": np.round(prec, 4),
        "Recall (Sensitivity)": np.round(rec, 4),
        "F1-Score": np.round(f1, 4),
        "Support": sup
    })
    return df


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    normalize: bool = True,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots a clinical confusion matrix with adjacent boundary severity analysis.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=DR_CLASSES,
        yticklabels=DR_CLASSES,
        cbar=True
    )
    plt.title(f"{model_name} — Confusion Matrix ({'Normalized' if normalize else 'Counts'})", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted DR Grade", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth DR Grade", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots One-vs-Rest ROC curves and Precision-Recall curves for all 5 DR classes.
    """
    num_classes = y_prob.shape[1]
    y_one_hot = np.eye(num_classes)[y_true]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curves
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_one_hot[:, c], y_prob[:, c])
        auc_val = roc_auc_score(y_one_hot[:, c], y_prob[:, c])
        axes[0].plot(fpr, tpr, lw=2, label=f"{DR_CLASSES[c]} (AUC = {auc_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"{model_name} — One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curves
    for c in range(num_classes):
        prec, rec, _ = precision_recall_curve(y_one_hot[:, c], y_prob[:, c])
        ap_val = average_precision_score(y_one_hot[:, c], y_prob[:, c])
        axes[1].plot(rec, prec, lw=2, label=f"{DR_CLASSES[c]} (AP = {ap_val:.3f})")

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"{model_name} — Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_calibration_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots a Reliability Diagram comparing model confidence to observed accuracy.
    """
    ece, bin_accs, bin_confs, bin_counts = compute_expected_calibration_error(y_true, y_prob, n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    bin_centers = np.linspace(0.05, 0.95, n_bins)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=2)

    # Actual calibration curve
    ax.plot(bin_confs, bin_accs, "s-", color="crimson", lw=2, label=f"{model_name} (ECE = {ece:.4f})")
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.2, color="crimson", edgecolor="black")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name} — Reliability Calibration Diagram", fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
