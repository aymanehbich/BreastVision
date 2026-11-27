# Automated Breast Cancer Classification in Mammography Images via Transfer Learning

Master ISI S3 – Deep Learning project (Projet 7), 2025-2026.
Compares feature extraction, partial fine-tuning and full fine-tuning of pre-trained
Xception, ResNeXt-101, EfficientNetV2 and MaxViT-Base, against a CNN trained from scratch,
with Grad-CAM interpretability. Task: benign vs. malignant on full mammograms (CBIS-DDSM).

> Status: Phase 0 (setup) finished, Phase 1-2 in progress. Interfaces between team members: [CONTRACT.md](CONTRACT.md).

## Setup (Windows)
`torch` and `torchvision` are **not** in the requirements files because the CPU and CUDA builds differ.
Follow these steps in order.

**1. Create the venv** with Python **3.12** (the only supported version):
```powershell
py -3.12 -m venv .venv
```

**2. Install torch and torchvision**, using **one** of the two commands below. Everyone uses the same versions:
torch **2.14.1** and torchvision **0.29.1**.

CPU only:
```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```
NVIDIA GPU:
```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/<cuXXX>
```
> **Note:** if cu130 fails or does not support your GPU, use cu126; each person can use a different CUDA build as long
> as the version numbers match.

For `<cuXXX>`, run `nvidia-smi` to check your driver and the highest CUDA version it supports, then take the exact GPU
command from [pytorch.org](https://pytorch.org/get-started/locally/). Keep the team's version numbers, and check that
the CUDA build you choose provides them.

**3. Install the other packages**, at the exact versions in the lock file:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

**4. Verify:**
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
On a GPU machine it **must** print `True`. `configs/base.yaml` uses `device: auto` (CUDA if available, else CPU).
`requirements.txt` lists the direct dependencies, without versions.

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
- `scripts/` – splits, leakage check, benchmark (GPU/CPU batch time, max batch in VRAM), results aggregation
- `tests/`, `notebooks/` – tests, EDA and results analysis
- `results/` – `manifest.csv`, `runs/<model>_<mode>_f<fold>/`, `tuning/`, `figures/`
- `team/<name>/PLAN.md` – personal plans
- `paper/` – English paper

## Team contributions
To be filled at the end of the project.
