# Automated Breast Cancer Classification in Mammography Images via Transfer Learning

Master ISI S3 – Deep Learning project (Projet 7), 2025-2026.
Compares feature extraction, partial fine-tuning and full fine-tuning of pre-trained
Xception, ResNeXt-101, EfficientNetV2 and MaxViT-Base, against a CNN trained from scratch,
with Grad-CAM interpretability.

> Status: Phase 0 (setup). See [PLAN.md](PLAN.md) for the full roadmap.

## Setup (Windows, CPU)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Dataset: Kaggle `hayder17/breast-cancer-detection`. Put your `kaggle.json` in
`%USERPROFILE%\.kaggle\`, then:
```powershell
.\.venv\Scripts\kaggle.exe datasets download -d hayder17/breast-cancer-detection -p data --unzip
```

## Layout
- `configs/base.yaml` – shared hyperparameters (same for all models)
- `src/` – data, models, training, evaluation, Grad-CAM, cross-validation
- `notebooks/` – EDA and results analysis
- `results/` – metrics, figures
- `paper/` – English paper

## Team contributions
To be filled at the end of the project.
