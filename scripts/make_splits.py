"""Rebuild results/manifest.csv: full mammograms + patient-level split + CV folds (owner: Aymane).

    python scripts/make_splits.py [--config configs/base.yaml]

Deterministic (seed in the config). Same as `python -m src.data build`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data import _stats, add_cv_folds, build_manifest, make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
sp = cfg["data"]["split"]
m = make_splits(build_manifest(cfg["data"]["root"]), cfg["seed"], (sp["train"], sp["val"], sp["test"]))
m = add_cv_folds(m, cfg["cv"]["folds"], cfg["seed"])
out = Path(cfg["data"]["manifest"])
out.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(out, index=False)
_stats(m)
print("wrote", out)
