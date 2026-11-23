"""Shared helpers: project paths, config loading, seeding, hardware description.

Everyone imports these instead of re-implementing them, so all runs behave the same way.
"""
from __future__ import annotations

import os
import platform
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "base.yaml"
RUNS_DIR = ROOT / "results" / "runs"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Read the shared YAML protocol (identical for every model and regime)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (torch is optional so data-only scripts stay light)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device(cfg: dict | None = None) -> str:
    """Resolve cfg['train']['device']: 'auto' -> 'cuda' if available else 'cpu'."""
    import torch
    want = (cfg or {}).get("train", {}).get("device", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def hardware_info(cfg: dict | None = None) -> dict:
    """Machine description stored in every run's meta.json.

    Training times are only comparable on the same hardware, so every run records: GPU name,
    VRAM, CUDA version, torch version, device, amp and grad_accum (see CONTRACT.md).
    """
    info = {"machine": platform.node(), "os": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version()}
    t = (cfg or {}).get("train", {})
    info.update(amp=t.get("amp"), grad_accum=t.get("grad_accum"))
    try:
        import torch
        device = get_device(cfg)
        info.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), device=device,
                    cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                    gpu_name=None, gpu_vram_gb=None)
        if device.startswith("cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.device(device))
            info.update(gpu_name=props.name, gpu_vram_gb=round(props.total_memory / 1024**3, 2))
    except ImportError:
        pass
    return info


def run_dir(model: str, mode: str, fold: int | None = None) -> Path:
    """results/runs/<model>_<mode>_f<fold>/ (fold = 'none' for the fixed train/val split)."""
    return RUNS_DIR / f"{model}_{mode}_f{'none' if fold is None else fold}"
