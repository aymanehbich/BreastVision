"""Stratified patient-level k-fold driver (owner: Fatima). Stub: signatures are fixed by CONTRACT.md."""
from __future__ import annotations

def run_cv(model_name: str, mode: str, cfg: dict):
    """Call train.run for every fold 0..cv.folds-1 and aggregate mean +/- std."""
    raise NotImplementedError
