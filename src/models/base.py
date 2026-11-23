"""Freeze policies shared by all models (owner: Houcine).

regime "extract": backbone frozen, only head_params trained
regime "partial": head + the last stage (model-specific `unfreeze_partial`)
regime "full"   : everything trainable
"""
from __future__ import annotations


def apply_regime(model, module, mode: str) -> None:
    """Set requires_grad on `model` according to mode ('extract' | 'partial' | 'full').

    `module` is the model file (e.g. src.models.maxvit) providing head_params / unfreeze_partial.
    TODO (Houcine): implement and test that trainable params extract << partial < full.
    """
    raise NotImplementedError


def count_params(model) -> tuple[int, int]:
    """Return (total, trainable) parameter counts, stored in meta.json."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
