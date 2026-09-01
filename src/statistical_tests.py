"""
Statistical Significance Testing and Bootstrap Validation Suite.
Calculates 95% Confidence Intervals via non-parametric bootstrapping for Macro F1, Accuracy, and QWK.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes empirical point estimate and (lower, upper) confidence intervals for Accuracy, Macro F1, and QWK.
    """
    rng = np.random.RandomState(random_seed)
    n_samples = len(y_true)

    boot_acc = []
    boot_macro_f1 = []
    boot_qwk = []

    # Point estimates
    point_acc = accuracy_score(y_true, y_pred)
    _, _, point_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    point_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        # Check for single-class draw
        if len(np.unique(sample_true)) < 2:
            continue

        acc = accuracy_score(sample_true, sample_pred)
        _, _, f1, _ = precision_recall_fscore_support(sample_true, sample_pred, average="macro", zero_division=0)
        qwk = cohen_kappa_score(sample_true, sample_pred, weights="quadratic")

        boot_acc.append(acc)
        boot_macro_f1.append(f1)
        boot_qwk.append(qwk)

    alpha = 1.0 - confidence_level
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    results = {
        "Accuracy": (
            round(float(point_acc), 4),
            round(float(np.percentile(boot_acc, lower_p)), 4),
            round(float(np.percentile(boot_acc, upper_p)), 4)
        ),
        "Macro F1": (
            round(float(point_f1), 4),
            round(float(np.percentile(boot_macro_f1, lower_p)), 4),
            round(float(np.percentile(boot_macro_f1, upper_p)), 4)
        ),
        "QWK": (
            round(float(point_qwk), 4),
            round(float(np.percentile(boot_qwk, lower_p)), 4),
            round(float(np.percentile(boot_qwk, upper_p)), 4)
        )
    }

    return results


def compare_models_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Weighted ResNet-18",
    model_b_name: str = "Attention Fusion",
    n_bootstraps: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Performs paired bootstrap resampling to compute p-values and difference distributions (Delta).
    """
    rng = np.random.RandomState(random_seed)
    n_samples = len(y_true)

    delta_acc = []
    delta_f1 = []
    delta_qwk = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        st = y_true[idx]
        spa = y_pred_a[idx]
        spb = y_pred_b[idx]

        if len(np.unique(st)) < 2:
            continue

        acc_a = accuracy_score(st, spa)
        acc_b = accuracy_score(st, spb)
        delta_acc.append(acc_b - acc_a)

        _, _, f1_a, _ = precision_recall_fscore_support(st, spa, average="macro", zero_division=0)
        _, _, f1_b, _ = precision_recall_fscore_support(st, spb, average="macro", zero_division=0)
        delta_f1.append(f1_b - f1_a)

        qwk_a = cohen_kappa_score(st, spa, weights="quadratic")
        qwk_b = cohen_kappa_score(st, spb, weights="quadratic")
        delta_qwk.append(qwk_b - qwk_a)

    records = [
        {
            "Metric": "Macro F1",
            "Mean Difference (B - A)": np.mean(delta_f1),
            "95% CI Lower": np.percentile(delta_f1, 2.5),
            "95% CI Upper": np.percentile(delta_f1, 97.5),
            "P-Value (Two-Sided)": 2.0 * min(np.mean(np.array(delta_f1) <= 0), np.mean(np.array(delta_f1) >= 0))
        },
        {
            "Metric": "Accuracy",
            "Mean Difference (B - A)": np.mean(delta_acc),
            "95% CI Lower": np.percentile(delta_acc, 2.5),
            "95% CI Upper": np.percentile(delta_acc, 97.5),
            "P-Value (Two-Sided)": 2.0 * min(np.mean(np.array(delta_acc) <= 0), np.mean(np.array(delta_acc) >= 0))
        },
        {
            "Metric": "QWK",
            "Mean Difference (B - A)": np.mean(delta_qwk),
            "95% CI Lower": np.percentile(delta_qwk, 2.5),
            "95% CI Upper": np.percentile(delta_qwk, 97.5),
            "P-Value (Two-Sided)": 2.0 * min(np.mean(np.array(delta_qwk) <= 0), np.mean(np.array(delta_qwk) >= 0))
        }
    ]

    return pd.DataFrame(records)
