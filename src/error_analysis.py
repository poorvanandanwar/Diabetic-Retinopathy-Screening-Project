"""
Clinical Error Analysis and Failure Case Diagnostics for DR Screening.
Identifies high-confidence misclassifications, critical false negatives, and adjacent grade errors.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .visualization import apply_clahe, overlay_heatmap, DR_CLASSES


def identify_clinical_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    image_paths: List[str],
    confidence_threshold: float = 0.70
) -> pd.DataFrame:
    """
    Categorizes and filters diagnostic errors into high-confidence mistakes,
    critical under-diagnoses (False Negatives on Referable DR), and adjacent grade errors.
    """
    confidences = np.max(y_prob, axis=1)
    is_error = (y_true != y_pred)

    records = []
    for idx in range(len(y_true)):
        t_lbl = int(y_true[idx])
        p_lbl = int(y_pred[idx])
        conf = float(confidences[idx])
        path = image_paths[idx]

        if not is_error[idx]:
            continue

        # Severity jump
        severity_delta = p_lbl - t_lbl  # negative: under-diagnosed, positive: over-diagnosed

        # Error taxonomy
        if t_lbl >= 2 and p_lbl <= 1:
            error_category = "Critical False Negative (Referable Missed)"
        elif t_lbl == 0 and p_lbl >= 2:
            error_category = "Severe False Positive"
        elif abs(severity_delta) == 1:
            error_category = "Adjacent Boundary Discrepancy"
        else:
            error_category = "Multi-Grade Discordance"

        is_high_confidence = conf >= confidence_threshold

        records.append({
            "sample_index": idx,
            "image_path": path,
            "true_grade": t_lbl,
            "true_name": DR_CLASSES[t_lbl],
            "pred_grade": p_lbl,
            "pred_name": DR_CLASSES[p_lbl],
            "confidence": round(conf, 4),
            "severity_delta": severity_delta,
            "error_category": error_category,
            "high_confidence_error": is_high_confidence
        })

    error_df = pd.DataFrame(records)
    return error_df


def plot_diagnostic_error_case(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    error_info: dict,
    model_name: str = "Attention Fusion",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Renders an in-depth diagnostic failure inspection sheet showing original, CLAHE,
    Grad-CAM heatmap, and superimposed view with error details and suspected clinical etiology.
    """
    clahe_rgb = apply_clahe(image_rgb)
    _, overlay = overlay_heatmap(image_rgb, cam)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    true_str = error_info["true_name"]
    pred_str = error_info["pred_name"]
    conf = error_info["confidence"]
    cat = error_info["error_category"]

    fig.suptitle(
        f"Diagnostic Error Investigation [{model_name}]\n"
        f"Ground Truth: {true_str} | Predicted: {pred_str} (Conf: {conf:.1%}) | Category: {cat}",
        fontsize=13,
        fontweight="bold",
        color="crimson"
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Fundus Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(clahe_rgb)
    axes[1].set_title("CLAHE Enhanced View", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM Activation (Model Focus)", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Superimposed Failure Overlay", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
