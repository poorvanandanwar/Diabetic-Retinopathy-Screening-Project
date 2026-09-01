"""
Visualization and Explainability Suite for Retinal Fundus Diabetic Retinopathy.
Provides CLAHE enhancement, Grad-CAM overlays, comparison grids, and lesion focus plots.
Supports OpenCV with graceful pure-PIL/Matplotlib fallback.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

DR_CLASSES = [
    "No DR (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Proliferative DR (4)"
]

DR_CLASS_SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def denormalize_image(
    tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Converts a normalized PyTorch image tensor (C, H, W) or (1, C, H, W) to RGB uint8 array (H, W, 3).
    """
    if hasattr(tensor, "dim") and tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if hasattr(tensor, "clone"):
        img = tensor.clone().detach().cpu().numpy()
    else:
        img = np.array(tensor)
    img = np.transpose(img, (1, 2, 0))
    img = (img * np.array(std) + np.array(mean)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in LAB color space.
    Enhances visibility of retinal lesions (microaneurysms, hemorrhages, hard/soft exudates).
    """
    if HAS_CV2:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        # Fallback using PIL equalization on lightness
        pil_img = Image.fromarray(image_rgb)
        pil_hsv = pil_img.convert("HSV")
        h, s, v = pil_hsv.split()
        v_eq = ImageOps.equalize(v)
        enhanced = Image.merge("HSV", (h, s, v_eq)).convert("RGB")
        return np.array(enhanced)


def overlay_heatmap(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays a normalized Grad-CAM map onto an RGB fundus image.

    Args:
        image_rgb (np.ndarray): Original RGB image of shape (H, W, 3), dtype uint8.
        cam (np.ndarray): Normalized Grad-CAM heatmap of shape (H, W) in [0, 1].
        alpha (float): Transparency factor for heatmap overlay (0.0 to 1.0).
        colormap (int, optional): OpenCV colormap.

    Returns:
        heatmap_colored (np.ndarray): Colored heatmap (H, W, 3) in RGB uint8.
        superimposed (np.ndarray): Blended RGB overlay (H, W, 3) in uint8.
    """
    cam_clipped = np.clip(cam, 0, 1)

    if HAS_CV2:
        cm = colormap if colormap is not None else cv2.COLORMAP_JET
        cam_uint8 = np.uint8(255 * cam_clipped)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cm)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Fallback using matplotlib colormap
        cmap = plt.get_cmap("jet")
        heatmap_rgb = (cmap(cam_clipped)[:, :, :3] * 255).astype(np.uint8)

    superimposed = np.float32(image_rgb) * (1.0 - alpha) + np.float32(heatmap_rgb) * alpha
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return heatmap_rgb, superimposed


def plot_single_explanation(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    true_label: int,
    pred_label: int,
    confidence: float,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots a 4-panel visual explanation: Original, CLAHE Enhanced, Heatmap, Superimposed Overlay.
    """
    clahe_rgb = apply_clahe(image_rgb)
    heatmap_rgb, superimposed = overlay_heatmap(image_rgb, cam)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    is_correct = (true_label == pred_label)
    status_text = "CORRECT" if is_correct else "MISCLASSIFIED"
    status_color = "green" if is_correct else "red"

    fig.suptitle(
        f"{model_name} Explainability | True: {DR_CLASSES[true_label]} | "
        f"Pred: {DR_CLASSES[pred_label]} (Conf: {confidence:.2%}) [{status_text}]",
        fontsize=14,
        fontweight="bold",
        color=status_color
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Fundus", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(clahe_rgb)
    axes[1].set_title("CLAHE Enhanced (Lesions)", fontsize=12)
    axes[1].axis("off")

    im = axes[2].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM Activation", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(superimposed)
    axes[3].set_title("Superimposed Overlay", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_multi_model_comparison(
    image_rgb: np.ndarray,
    cams_dict: dict,
    true_label: int,
    pred_labels: dict,
    confidences: dict,
    attention_weights: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots a multi-model comparative explainability figure:
    Original Fundus, ResNet-18 CAM, EfficientNet-B0 CAM, and Attention-Fused CAM.
    """
    models_to_plot = list(cams_dict.keys())
    n_models = len(models_to_plot)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4.5 * (n_models + 1), 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original Fundus\nTrue: {DR_CLASSES[true_label]}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for idx, name in enumerate(models_to_plot):
        ax = axes[idx + 1]
        cam = cams_dict[name]
        _, overlay = overlay_heatmap(image_rgb, cam)
        pred = pred_labels.get(name, true_label)
        conf = confidences.get(name, 0.0)

        extra_info = ""
        if name.lower().startswith("attention") and attention_weights:
            extra_info = f"\n(Attn: Res={attention_weights[0]:.2f}, Eff={attention_weights[1]:.2f})"

        ax.imshow(overlay)
        ax.set_title(
            f"{name}\nPred: {DR_CLASSES[pred]} ({conf:.1%}){extra_info}",
            fontsize=10,
            fontweight="bold" if pred == true_label else "normal",
            color="darkgreen" if pred == true_label else "crimson"
        )
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_grade_progression_grid(
    samples_per_class: List[dict],
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots a 5x4 grid demonstrating Grad-CAM explainability across the entire DR severity spectrum:
    Grade 0 (No DR) -> Grade 1 (Mild) -> Grade 2 (Moderate) -> Grade 3 (Severe) -> Grade 4 (PDR).
    """
    fig, axes = plt.subplots(5, 4, figsize=(18, 22))

    for grade_idx, sample in enumerate(samples_per_class):
        img_rgb = sample["image_rgb"]
        clahe_rgb = apply_clahe(img_rgb)
        cam = sample["cam"]
        _, overlay = overlay_heatmap(img_rgb, cam)

        true_lbl = sample["true_label"]
        pred_lbl = sample["pred_label"]
        conf = sample["confidence"]
        pathology_desc = sample.get("description", "")

        axes[grade_idx, 0].imshow(img_rgb)
        axes[grade_idx, 0].set_title(f"Grade {grade_idx}: {DR_CLASSES[grade_idx]}\n(Original Fundus)", fontsize=11)
        axes[grade_idx, 0].axis("off")

        axes[grade_idx, 1].imshow(clahe_rgb)
        axes[grade_idx, 1].set_title(f"CLAHE Enhanced\n{pathology_desc}", fontsize=10)
        axes[grade_idx, 1].axis("off")

        im = axes[grade_idx, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[grade_idx, 2].set_title(f"Grad-CAM Activation Map", fontsize=11)
        axes[grade_idx, 2].axis("off")

        axes[grade_idx, 3].imshow(overlay)
        axes[grade_idx, 3].set_title(f"Overlay | Pred: {DR_CLASSES[pred_lbl]} ({conf:.1%})", fontsize=11, fontweight="bold")
        axes[grade_idx, 3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
