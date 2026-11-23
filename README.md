# Automated Breast Cancer Classification in Mammography Images via Transfer Learning

Master ISI S3 – Deep Learning project (Projet 7), 2025-2026.
Compares feature extraction, partial fine-tuning and full fine-tuning of pre-trained
Xception, ResNeXt-101, EfficientNetV2 and MaxViT-Base, against a CNN trained from scratch,
with Grad-CAM interpretability. Task: benign vs. malignant on full mammograms (CBIS-DDSM).

> Status: Phase 0 (setup) finished, Phase 1-2 in progress. Interfaces between team members: [CONTRACT.md](CONTRACT.md).

## Setup (Windows, CPU only)
Python 3.11 or 3.12 recommended.
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
`requirements-lock.txt` lists the exact versions used by the project.

## Dataset
Kaggle `awsaf49/cbis-ddsm-breast-cancer-image-dataset` (folders `csv/` and `jpeg/`). Put your `kaggle.json` in
`%USERPROFILE%\.kaggle\`, then:
```powershell
.\.venv\Scripts\kaggle.exe datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p data/cbis-ddsm --unzip
```
Only the 3,103 **full mammograms** are used (1,728 benign / 1,375 malignant, 1,566 patients).
Split by patient, never by image: 2,213 train / 445 val / 445 test, plus 5 patient-level CV folds on train+val.
The split is already committed in `results/manifest.csv`.

## Commands (run from the project root)
```powershell
.\.venv\Scripts\python.exe -m src.data cache         # resize images once to data/cache/224 (slow, run once)
.\.venv\Scripts\python.exe scripts/make_splits.py    # rebuild results/manifest.csv (deterministic, seed 42)
.\.venv\Scripts\python.exe scripts/check_leakage.py --hash   # no patient / image shared between splits
.\.venv\Scripts\python.exe tests/test_data.py        # smoke test of the data interface
```
**No dataset yet?** Set `data.dummy: true` in `configs/base.yaml`: `get_loaders` then returns random
images with the right shapes, so you can develop and test your code.

## Layout
- `configs/base.yaml` – shared protocol, identical for all models
- `src/` – `common.py`, `data.py`, `models/` (one file per model), `train.py`, `evaluate.py`, `gradcam.py`, `cv.py`
- `scripts/` – splits, leakage check, CPU benchmark, results aggregation
- `tests/`, `notebooks/` – tests, EDA and results analysis
- `results/` – `manifest.csv`, `runs/<model>_<mode>_f<fold>/`, `tuning/`, `figures/`
- `team/<name>/PLAN.md` – personal plans
- `paper/` – English paper

## Team contributions
To be filled at the end of the project.
