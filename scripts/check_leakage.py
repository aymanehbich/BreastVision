"""Check the manifest for data leakage (owner: Aymane). Exit code 1 if a check fails.

    python scripts/check_leakage.py [--hash]

Checks: no patient in two splits; no patient in two CV folds; test rows are never in a fold;
with --hash, no identical JPEG content shared between different splits (slow, reads all files).
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--hash", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
m = pd.read_csv(cfg["data"]["manifest"])
errors = []

n = m.groupby("patient_id")["split"].nunique()
if (n > 1).any():
    errors.append(f"{(n > 1).sum()} patients appear in several splits")
pool = m[m["split"] != "test"]
n = pool.groupby("patient_id")["fold"].nunique()
if (n > 1).any():
    errors.append(f"{(n > 1).sum()} patients appear in several CV folds")
if ((m["split"] == "test") & (m["fold"] != -1)).any():
    errors.append("test images assigned to a CV fold")
if (pool["fold"] < 0).any():
    errors.append("train/val images without a CV fold")

if a.hash:
    root = Path(cfg["data"]["root"]) / "jpeg"
    m["md5"] = [hashlib.md5((root / j).read_bytes()).hexdigest() for j in m["jpg"]]
    d = m.groupby("md5")["split"].nunique()
    if (d > 1).any():
        errors.append(f"{(d > 1).sum()} identical images shared between splits")
    print("duplicate image files:", int(m.duplicated("md5").sum()))

print("images:", len(m), "| patients:", m["patient_id"].nunique())
print(pd.crosstab(m["split"], m["label"], margins=True).to_string())
if errors:
    print("LEAKAGE CHECK FAILED:", *errors, sep="\n - ")
    sys.exit(1)
print("leakage check passed")
