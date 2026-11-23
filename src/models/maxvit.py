"""MaxViT-Base (owner: Aymane).

Interface required by CONTRACT.md. Keep these four function names and signatures.
"""
from __future__ import annotations


def build(num_classes: int, timm_name: str):
    """Return the model with a fresh num_classes head (timm maxvit_base_tf_224, ImageNet weights; also needs reshape_transform for Grad-CAM)."""
    raise NotImplementedError


def head_params(model):
    """Parameters of the classification head only (trained in every regime)."""
    raise NotImplementedError


def unfreeze_partial(model) -> None:
    """Make the last stage trainable (regime 'partial'); the rest stays frozen."""
    raise NotImplementedError


def gradcam_target_layers(model) -> list:
    """Layer(s) used as Grad-CAM target (last conv / last transformer block)."""
    raise NotImplementedError


def reshape_transform(tensor):
    """Turn MaxViT token output into (B, C, H, W) for pytorch-grad-cam."""
    raise NotImplementedError
