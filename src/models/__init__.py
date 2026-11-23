"""Model package: one file per model, all exposing the same interface (see CONTRACT.md).

Usage:
    from src.models import get_module, build_model
    model = build_model("maxvit", cfg)
"""
from __future__ import annotations

import importlib

# config name (cfg["models"]) -> module in this package
MODULES = {
    "xception": "xception",          # Fatima
    "resnext101": "resnext",         # Houcine
    "efficientnetv2": "efficientnet",  # Houcine
    "maxvit": "maxvit",              # Aymane
    "scratch_cnn": "scratch_cnn",    # Aymane
}


def get_module(name: str):
    """Return the module implementing build / unfreeze_partial / head_params / gradcam_target_layers."""
    if name not in MODULES:
        raise KeyError(f"unknown model '{name}', expected one of {sorted(MODULES)}")
    return importlib.import_module(f"{__name__}.{MODULES[name]}")


def build_model(name: str, cfg: dict, num_classes: int = 2):
    """Create the model `name` with its ImageNet weights (random init for scratch_cnn)."""
    return get_module(name).build(num_classes, cfg["models"][name])
