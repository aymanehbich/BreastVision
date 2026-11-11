"""Data pipeline for CBIS-DDSM (awsaf49 Kaggle copy: csv/ + jpeg/).

Steps
1. build_manifest : one row per FULL mammogram (3,103 expected), label rule applied.
2. make_splits    : stratified, PATIENT-LEVEL train/val/test split (no patient in two splits).
3. cache_images   : one-time resize (letterbox to square) so training never decodes 4000+ px JPEGs.
4. MammoDataset   : torch Dataset reading the cached images + augmentation/normalisation.

CLI (run from the project root):
    python -m src.data build     # writes results/manifest.csv and prints dataset statistics
    python -m src.data cache     # writes data/cache/<size>/*.png  (slow, run once)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = ["benign", "malignant"]
Image.MAX_IMAGE_PIXELS = None  # full mammograms are large but trusted


# --------------------------------------------------------------------------- manifest
def build_manifest(root: str | Path) -> pd.DataFrame:
    """Return one row per full-mammogram image with a binary label.

    The case CSVs list one row per *abnormality*; 'image file path' is the FULL mammogram.
    We map it to a JPEG through the series UID in dicom_info.csv. We do NOT filter on
    SeriesDescription == 'full mammogram images': 246 full images have a blank description.
    Label rule: BENIGN and BENIGN_WITHOUT_CALLBACK -> benign (0); an image is malignant (1)
    if ANY of its lesions is MALIGNANT.
    """
    root = Path(root)
    di = pd.read_csv(root / "csv" / "dicom_info.csv")
    parts = []
    for kind in ("calc", "mass"):
        for split in ("train", "test"):
            d = pd.read_csv(root / "csv" / f"{kind}_case_description_{split}_set.csv")
            d.columns = [c.replace(" ", "_") for c in d.columns]
            parts.append(d.assign(kind=kind, official_split=split))
    cases = pd.concat(parts, ignore_index=True)
    cases["series_uid"] = cases["image_file_path"].str.strip().str.split("/").str[-2]
    cases["malignant"] = cases["pathology"].str.upper().eq("MALIGNANT")

    # series UID -> JPEG path (must be unique, otherwise the mapping is ambiguous)
    jpg = di[["SeriesInstanceUID", "image_path"]].drop_duplicates("SeriesInstanceUID")
    cases = cases.merge(jpg, left_on="series_uid", right_on="SeriesInstanceUID", how="left")
    if cases["image_path"].isna().any():
        raise ValueError(f"{cases['image_path'].isna().sum()} abnormalities without a JPEG")
    cases["jpg"] = cases["image_path"].str.replace("CBIS-DDSM/jpeg/", "", regex=False)

    m = (cases.groupby("series_uid")
         .agg(jpg=("jpg", "first"), patient_id=("patient_id", "first"),
              n_patients=("patient_id", "nunique"), n_lesions=("pathology", "size"),
              malignant=("malignant", "any"), all_malignant=("malignant", "all"),
              kind=("kind", lambda s: "+".join(sorted(set(s)))),
              breast_density=("breast_density", "first"),
              side=("left_or_right_breast", "first"), view=("image_view", "first"),
              official_split=("official_split", lambda s: "+".join(sorted(set(s)))))
         .reset_index().rename(columns={"series_uid": "image_id"}))
    assert (m["n_patients"] == 1).all(), "an image maps to several patients"
    m["label"] = m["malignant"].astype(int)
    m["label_conflict"] = m["malignant"] & ~m["all_malignant"]   # benign + malignant lesion on one image
    missing = [p for p in m["jpg"] if not (root / "jpeg" / p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} JPEGs missing, e.g. {missing[:2]}")
    return m.drop(columns=["n_patients", "all_malignant", "malignant"])


# --------------------------------------------------------------------------- splits
def make_splits(m: pd.DataFrame, seed: int = 42, frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Add a 'split' column (train/val/test). Grouped by patient, stratified by label.

    Fractions are approximate (whole patients move together). Test is carved out first,
    then validation from the remainder.
    """
    m = m.copy()
    k_test = round(1 / frac[2])
    rest_idx, test_idx = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=seed)
                              .split(m, m["label"], m["patient_id"]))
    rest = m.iloc[rest_idx]
    k_val = round((1 - frac[2]) / frac[1])
    tr_idx, val_idx = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=seed)
                           .split(rest, rest["label"], rest["patient_id"]))
    m["split"] = "train"
    m.loc[m.index[test_idx], "split"] = "test"
    m.loc[rest.index[val_idx], "split"] = "val"
    check_no_leakage(m)
    return m


def check_no_leakage(m: pd.DataFrame) -> None:
    """Assert that no patient appears in more than one split."""
    n = m.groupby("patient_id")["split"].nunique()
    assert (n == 1).all(), f"{(n > 1).sum()} patients appear in several splits"


# --------------------------------------------------------------------------- cache
def _letterbox(img: Image.Image, size: int) -> Image.Image:
    """Pad to a square (black) then resize, so the aspect ratio is preserved."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 0)
    canvas.paste(img, (0, 0))      # breast is left-aligned in most scans; pad on the right/bottom
    return canvas.resize((size, size), Image.BILINEAR)


def _cache_one(args):
    src, dst, size = args
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    _letterbox(Image.open(src).convert("L"), size).save(dst)


def cache_images(m: pd.DataFrame, root: Path, cache_dir: Path, size: int, workers: int = 4) -> None:
    jobs = [(root / "jpeg" / j, cache_dir / f"{i}.png", size) for i, j in zip(m["image_id"], m["jpg"])]
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(_cache_one, jobs, chunksize=8), 1):
            if k % 200 == 0:
                print(f"cached {k}/{len(jobs)}", flush=True)


# --------------------------------------------------------------------------- torch dataset
def get_transforms(cfg: dict, train: bool):
    import torchvision.transforms as T
    d, a = cfg["data"], cfg["augmentation"]
    ops = []
    if train:
        ops += [T.RandomHorizontalFlip(a["hflip"]),
                T.RandomAffine(a["rotation_deg"], scale=tuple(a["scale"])),
                T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"])]
    ops += [T.ToTensor(), T.Normalize(d["imagenet_mean"], d["imagenet_std"])]
    return T.Compose(ops)


class MammoDataset:
    """Reads cached letterboxed images; replicates the grayscale channel to 3 for ImageNet models."""

    def __init__(self, manifest: pd.DataFrame, cache_dir: str | Path, transform=None):
        self.m = manifest.reset_index(drop=True)
        self.cache_dir, self.transform = Path(cache_dir), transform

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        r = self.m.iloc[i]
        img = Image.open(self.cache_dir / f"{r['image_id']}.png").convert("RGB")
        return (self.transform(img) if self.transform else img), int(r["label"])


# --------------------------------------------------------------------------- CLI
def _stats(m: pd.DataFrame) -> None:
    print(f"images: {len(m)} | patients: {m['patient_id'].nunique()}")
    print("label (0=benign, 1=malignant):", m["label"].value_counts().sort_index().to_dict())
    print("images with >1 lesion:", int((m["n_lesions"] > 1).sum()),
          "| conflicting benign+malignant lesions:", int(m["label_conflict"].sum()))
    print("breast density:", m["breast_density"].value_counts().sort_index().to_dict())
    if "split" in m:
        print(pd.crosstab(m["split"], m["label"], margins=True).to_string())
        print("patients per split:", m.groupby("split")["patient_id"].nunique().to_dict())
    off = m["official_split"].str.contains(r"\+").sum()
    print("images listed in both official train and test CSV:", int(off))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "cache"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    root = Path(cfg["data"]["root"])
    man_path = Path(cfg["data"]["manifest"])
    if a.cmd == "build":
        man = make_splits(build_manifest(root), cfg["seed"],
                          tuple(cfg["data"]["split"][k] for k in ("train", "val", "test")))
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man.to_csv(man_path, index=False)
        _stats(man)
        print("wrote", man_path)
    else:
        man = pd.read_csv(man_path)
        size = cfg["data"]["image_size"]
        cache_images(man, root, Path(cfg["data"]["cache_dir"]) / str(size), size, a.workers)
        print("done")
