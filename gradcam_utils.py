"""
Grad-CAM utilities for the dual-input MultiBackboneFundusModel.
Generates heatmaps for both EfficientNet (full image) and ResNet50 (OD ROI) branches.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def _get_gradcam_map(model, branch, target_layer, input_tensor, class_idx=None):
    """
    Core Grad-CAM implementation that works with dual-input models.
    Hooks into the specified target_layer of the given branch.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor[0], input_tensor[1])

    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    score = output[0, class_idx]
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    act = activations[0].squeeze(0)       # (C, H, W)
    grad = gradients[0].squeeze(0)        # (C, H, W)

    weights = grad.mean(dim=(1, 2))       # (C,)
    cam = (weights[:, None, None] * act).sum(0)  # (H, W)
    cam = F.relu(cam)

    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, class_idx


def generate_gradcam_overlay(model, input_tensor_pair, target_layer, raw_image_np,
                              class_idx=None, colormap=cv2.COLORMAP_JET):
    """
    Returns (cam_array 0-1, overlay_rgb uint8).
    raw_image_np: H×W×3 float32 image in [0,1] range.
    """
    cam, pred_class = _get_gradcam_map(model, None, target_layer, input_tensor_pair, class_idx)

    h, w = raw_image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = 0.5 * raw_image_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return cam_resized, overlay, pred_class


def prepare_input(image_np, transform, device):
    """Convert HxWx3 uint8 numpy image to model-ready tensor."""
    pil = Image.fromarray(image_np.astype(np.uint8))
    tensor = transform(pil).unsqueeze(0).to(device)
    return tensor
