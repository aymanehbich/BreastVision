# Personal plan: Fatima Ezzahra

Role: evaluation, cross-validation driver, Grad-CAM, figures, final assembly, Xception. Shared overview: `PLAN.md`. Interfaces: `CONTRACT.md`.

## Code I own
`src/evaluate.py`, `src/gradcam.py`, `src/cv.py`, `src/models/xception.py`, `scripts/aggregate_results.py`, `tests/test_eval.py`, `tests/fake_runs.py`, `notebooks/02_results.ipynb`, `results/figures/`.

## Phase 0 (days 1-2)
- [ ] Python 3.11/3.12 venv; install the CUDA build of `torch` (commands in the README) and check that `torch.cuda.is_available()` is `True`; `pip install -r requirements.txt`; clone the repo
- [ ] Smoke test Xception on dummy data: create, forward pass, parameter count, time of 1 batch forward+backward. Check the exact timm name and pretrained weights (`xception41` or `xception`) and tell the team, so `configs/base.yaml` can be fixed
- [ ] Read `CONTRACT.md` with the others (1 hour) and confirm your GPU model and VRAM (`nvidia-smi`)

## Phase 2: evaluation code on fake data (days 3-7)
- `xception.py`
- `tests/fake_runs.py`: fake run folders following the run-folder format of `CONTRACT.md`, so I can work before real runs exist
- `evaluate.py`: accuracy, precision, recall, F1, AUC, confusion matrix, ROC, learning curves; tested on a hand-computed example
- `scripts/aggregate_results.py`: global table (Model | Regime | Accuracy | Precision | Recall | F1 | AUC | #Params | Training time)
- `cv.py`: patient-level stratified k-fold driver using the `fold` column of `results/manifest.csv`, mean and standard deviation
- `gradcam.py` with `pytorch-grad-cam`, using each model's `gradcam_target_layers` (and MaxViT's `reshape_transform`)

## Phase 3: experiments (days 7-14)
Xception in 3 regimes under the common protocol; CV for extract and partial regimes.

## Phase 4: evaluation and interpretability (days 14-19), I consolidate
- Collect everyone's run folders, run the aggregation, select the best model per regime on validation (and CV mean and std)
- Run the test set once (`final=True`) for the selected models, with each model's owner
- Final table, ROC curves for all models on one plot, confusion matrices, Grad-CAM for the best model and one comparator (correct and wrong predictions)
- Careful wording: heatmaps show where the model looks, not proven lesion detection

## Paper (days 10-20)
III.7 (metrics with formulas), IV.3-IV.7 (confusion matrix, ROC, AUC, global table, comparison), V Interpretability, VII Conclusion. Numbers only from `results/` files.
