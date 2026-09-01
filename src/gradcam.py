"""
Grad-CAM and Grad-CAM++ Engine for Retinal Fundus Diabetic Retinopathy Models.
Supports ResNet-18, EfficientNet-B0, and Attention Fusion architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    Computes class activation maps for a target convolutional layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device = None):
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
        return_logits: bool = False
    ):
        """
        Generate Grad-CAM heatmap for a given input tensor and target class.

        Args:
            input_tensor (torch.Tensor): Image tensor of shape (1, C, H, W).
            target_class (int, optional): Target class index. If None, uses model's predicted class.
            return_logits (bool): Whether to return output logits.

        Returns:
            cam (np.ndarray): Normalized Grad-CAM heatmap in [0, 1] of shape (H, W).
            target_class (int): The class index evaluated.
            logits (torch.Tensor, optional): Output logits if return_logits=True.
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Gradients: (1, K, H', W'), Activations: (1, K, H', W')
        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise RuntimeError(
                "No activations/gradients captured — the target_layer never fired during "
                "this forward/backward pass. Verify target_layer belongs to the model actually "
                "being called in generate_cam()."
            )

        # Global average pooling of gradients over spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Linear combination of weighted activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if return_logits:
            return cam, target_class, logits.detach()
        return cam, target_class

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.
    Provides better localization for multiple occurrences of object/lesion instances.
    """

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
        return_logits: bool = False
    ):
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients  # (1, K, H', W')
        activations = self.activations  # (1, K, H', W')

        # Compute Grad-CAM++ alpha weights
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        eps = 1e-8

        # sum over spatial dimensions
        spatial_sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2.0 * grad_2 + spatial_sum_activations * grad_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)

        alphas = grad_2 / alpha_denom  # (1, K, H', W')
        # Grad-CAM++ weights each spatial location by exp(score) as in the original paper.
        # Raw classification logits are unbounded, so exp() can overflow to inf/NaN;
        # clamp before exponentiating to keep this numerically stable.
        score_exp = torch.exp(torch.clamp(score.detach(), max=50.0))
        positive_gradients = F.relu(score_exp * gradients)
        weights = (alphas * positive_gradients).sum(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if return_logits:
            return cam, target_class, logits.detach()
        return cam, target_class


class MultiBranchAttentionGradCAM:
    """
    Grad-CAM Explainer for Attention Fusion Models.
    Computes branch-specific Grad-CAMs for ResNet and EfficientNet,
    and calculates an Attention-Weighted Combined Activation Map.
    """

    def __init__(
        self,
        fusion_model: nn.Module,
        resnet_target_layer: nn.Module,
        effnet_target_layer: nn.Module,
        device: torch.device = None
    ):
        self.model = fusion_model
        self.device = device or next(fusion_model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        self.resnet_cam = GradCAM(self.model, resnet_target_layer, self.device)
        self.effnet_cam = GradCAM(self.model, effnet_target_layer, self.device)

    def generate_cams(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None
    ):
        """
        Generate ResNet CAM, EfficientNet CAM, and Attention-Fused CAM.

        Returns:
            dict containing:
                'resnet_cam': (H, W) np.ndarray
                'effnet_cam': (H, W) np.ndarray
                'fused_cam': (H, W) np.ndarray
                'attention_weights': [w_resnet, w_effnet]
                'target_class': int
                'predicted_class': int
                'probabilities': (5,) np.ndarray
        """
        input_tensor = input_tensor.to(self.device)

        # Forward pass to get predictions and attention weights
        with torch.no_grad():
            output, attn_weights = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            pred_class = int(probs.argmax())

        if target_class is None:
            target_class = pred_class

        w_resnet = float(attn_weights[0, 0].item())
        w_effnet = float(attn_weights[0, 1].item())

        # Generate individual CAMs
        res_cam, _ = self.resnet_cam.generate_cam(input_tensor, target_class)
        eff_cam, _ = self.effnet_cam.generate_cam(input_tensor, target_class)

        # Attention-weighted fusion
        fused_cam = w_resnet * res_cam + w_effnet * eff_cam
        f_min, f_max = fused_cam.min(), fused_cam.max()
        if f_max - f_min > 1e-8:
            fused_cam = (fused_cam - f_min) / (f_max - f_min)
        else:
            fused_cam = np.zeros_like(fused_cam)

        return {
            "resnet_cam": res_cam,
            "effnet_cam": eff_cam,
            "fused_cam": fused_cam,
            "attention_weights": [w_resnet, w_effnet],
            "target_class": target_class,
            "predicted_class": pred_class,
            "probabilities": probs
        }

    def remove_hooks(self):
        self.resnet_cam.remove_hooks()
        self.effnet_cam.remove_hooks()
