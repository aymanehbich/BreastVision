"""Training loop for modes extract | partial | full (owner: Houcine). Stub: signatures are fixed by CONTRACT.md."""
from __future__ import annotations

def run(model, mode: str, cfg: dict, fold: int | None = None, final: bool = False):
    """Train, write results/runs/<model>_<mode>_f<fold>/ and return that path.

    Writes history.csv, val_preds.csv, meta.json (and test_preds.csv only if final=True).
    """
    raise NotImplementedError
