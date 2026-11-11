# Plan: Breast Cancer Classification in Mammograms via Transfer Learning (Projet 7)

## Context
Master ISI S3 Deep Learning project (Pr. Aatila). Deliverables: (1) original work, (2) Git repo with commented, organised code + README.md + regular commit history, (3) English paper following the PDF structure, (4) a contributions table per student (4 students). Due no later than the DL exam day; nothing accepted afterwards.
The project folder currently holds only the PDF (`cahier des charge/`). Machine: Windows, **no NVIDIA GPU**, Python 3.14, git 2.55. Decisions made: PyTorch + timm, 4 students, 3+ weeks, **CPU-only training** (see Risk 1).

## Spec traps to handle (from the PDF)
- Section "Contributions" mentions *Diabetic Retinopathy*: a copy-paste error from another project. The paper must say **breast cancer / mammography**.
- Required models: **Xception, ResNeXt-101, EfficientNetV2, MaxViT-Base**, plus a **CNN from scratch** baseline.
- Required regimes (title): **feature extraction, partial fine-tuning, full fine-tuning**.
- Required: same training conditions for all models, cross-validation, Grad-CAM, ROC/AUC/confusion matrices, params count + training time table, English paper, contributions table, GitHub link inside the paper.

## Key risks and how the plan handles them
1. **CPU only.** MaxViT-Base and ResNeXt-101 (~90-120M params) are slow to train on CPU. Mitigations, in order:
   - Use 224x224 inputs, small batch, `torch.set_num_threads`, and save a checkpoint every epoch.
   - **Feature extraction**: backbone frozen, so compute embeddings once per model, cache to disk, train only the head. This is cheap on CPU.
   - **Partial fine-tuning**: unfreeze only the last 1-2 stages; cache the frozen early-stage outputs where possible, or train on a stratified subset.
   - **Full fine-tuning**: costliest. First run a timing benchmark (1 batch fwd+bwd per model). If an epoch of a model exceeds ~1h, run full fine-tuning on a stratified subset with few epochs, and state this in the Limitations section.
   - Fallback (outside the "local only" choice, but worth knowing): Kaggle gives free GPU hours and hosts the dataset. Using it for full fine-tuning only would keep all conditions identical and save days. Decide after the benchmark in Phase 1.
2. **Python 3.14**: PyTorch/timm wheels may be missing. Create a venv with Python 3.11 or 3.12 (install via `py -3.12` or python.org) and pin versions in `requirements.txt`.
3. **Dataset: CBIS-DDSM, JPG version (chosen; the original hayder17 dataset has only 336 images and is dropped).** The user downloads it later into `data/cbis-ddsm/`. Counts are NOT verified: the Kaggle page lists 10,239 files, but these include full mammograms, crops and ROI masks. The real sample count comes from the CSVs (`calc_case(with_jpg_img).csv`, `mass_case(with_jpg_img).csv`) after download; fill the paper only with computed numbers. Rules:
   - Use **full mammogram images only** (`jpg_fullMammo_img_path`); crop/ROI images may be missing or mismatched.
   - Label: BENIGN and BENIGN_WITHOUT_CALLBACK -> benign; MALIGNANT -> malignant; an image is malignant if any of its lesions is malignant.
   - Cite the CBIS-DDSM (Lee et al. 2017, Sci Data) and TCIA (Clark et al. 2013) papers given on the Kaggle page.
   - Fallback if fewer than ~2,000 usable images or labels too messy: CMMD PNG (Kaggle `magpiesings/cmmd-mammogram-png-clinical-classification`).
4. **Data leakage.** One patient has several images/views. **Split by patient ID, never by image**, and assert no patient appears in two splits. Also hash-check for duplicate images.
5. **Metric choice.** Medical task: report recall/sensitivity and AUC, not just accuracy; handle class imbalance (class weights or weighted sampler).

## Repo layout (create in `c:\Users\hp\Desktop\S3\Deep Learning\projet`)
```
README.md            description, dependencies, how to run
requirements.txt
.gitignore           data/, checkpoints, venv, cached features
configs/base.yaml    single shared hyperparameter file (same for all models)
src/
  data.py            dataset, split, transforms (preprocess + augmentation)
  models.py          timm model factory (4 models) + CNN-from-scratch + freeze policies
  train.py           training loop (mode: extract | partial | full), logging, timing
  evaluate.py        metrics, confusion matrix, ROC, AUC, results CSV
  gradcam.py         Grad-CAM (pytorch-grad-cam; CNN target layers + MaxViT-compatible reshape)
  cv.py              stratified k-fold driver
notebooks/01_eda.ipynb  02_results.ipynb
results/             metrics CSV, curves, ROC, confusion matrices, gradcam figs
paper/               main.tex (or docx) + figures + references.bib
```
Git: `git init` first, then commit at each step below (regular history is graded).

## Phases (3 weeks, 4 students)

### Phase 0: setup (days 1-2), all
- `git init`, GitHub remote, `.gitignore`, shared repo, branch per person.
- Python 3.11/3.12 venv; install `torch` (CPU), `timm`, `scikit-learn`, `pandas`, `matplotlib`, `opencv-python`, `grad-cam`, `pyyaml`, `kaggle`.
- Download dataset; run duplicate check; write EDA notebook (class counts, image sizes, sample grid).
- Run CPU benchmark per model; fix image size, batch size, subset policy (Risk 1).
- **Freeze the protocol** in `configs/base.yaml`: seed, split, optimizer, LR, batch, epochs, early stopping, augmentation. Identical for all models.

### Phase 1: data pipeline (days 2-4), Student A
- Stratified train/val/test split (e.g. 70/15/15) with fixed seed; test set never touched until final evaluation.
- Preprocessing: resize to 224, 3-channel replication of grayscale, ImageNet mean/std normalisation (optional CLAHE, state if used).
- Augmentation (train only): flips, small rotation, brightness/contrast, light scale. No augmentation on val/test.
- Class imbalance handling per EDA.

### Phase 2: models and training code (days 3-7), Students B and C
- `models.py`: `timm.create_model` for `xception`, `resnext101_32x8d`, `tf_efficientnetv2_s` (or `efficientnetv2_rw_s`; pick one and name it exactly in the paper), `maxvit_base_tf_224`; replace head with new classifier (num_classes from data).
- CNN from scratch: ~4 conv blocks + BN + dropout, random init.
- Freeze policies: `extract` (all frozen except head), `partial` (last stage(s) + head), `full` (all).
- Training: AdamW (differential LR for backbone vs. head in fine-tuning), cosine or ReduceLROnPlateau, early stopping on val loss, log per-epoch train/val loss and accuracy, wall-clock time, trainable/total params.
- Smoke-test each model/mode on a tiny subset before real runs.

### Phase 3: experiments (days 7-14), Student B/C run, D monitors
- Matrix: 4 models x 3 regimes (+ scratch CNN) on the fixed split.
- Cross-validation: stratified 5-fold (3-fold if too slow) at least for feature-extraction and partial regimes; report mean ± std. State clearly which runs got CV.
- Small hyperparameter study (LR, unfreeze depth) on one or two models; log all in the results CSV.
- Each run saves: best checkpoint, history CSV, test predictions (probabilities), time.

### Phase 4: evaluation and interpretability (days 12-17), Student D
- Test metrics: accuracy, precision, recall, F1, AUC; confusion matrix; ROC curves (all models on one plot); learning curves.
- Global table exactly as the PDF: Model | Accuracy | Precision | Recall | F1 | AUC | #Params | Training time (add regime column).
- Grad-CAM for the best model and one comparator: correct vs. wrong predictions, malignant/benign examples. Be careful with claims: heatmaps show where the model looks, not proven lesion detection.
- Comparative analysis answering the PDF questions: most accurate, lightest, most robust (CV std, behaviour across regimes).

### Phase 5: paper (days 10-20, written in parallel), all
Write in English following the PDF order: Abstract (+ 5-6 keywords, GitHub link) → I Introduction (context, problem, contributions, outline) → II Related work (+ table Reference/Year/Dataset/Technique/Accuracy/Limits, positioning) → III Methodology (workflow figure, dataset, preprocessing, augmentation, models, training parameters, metrics with formulas) → IV Results (hardware/software, curves, confusion matrix, ROC, AUC, global table, comparison) → V Interpretability → VI Discussion (strengths, limits, perspectives) → VII Conclusion + contributions table + References.
- Related work: collect 10-15 real, citable papers and verify each one (authors, year, numbers) from the source; no invented references or accuracy figures. Add citations for every "(Références !)" bullet (cancer statistics, e.g. GLOBOCAN/WHO; BI-RADS variability; transfer learning; the four model papers).
- Fill result numbers only from `results/` files, never by hand.

### Phase 6: finalisation (days 19-21), all
- README: description, dependencies, exact commands to reproduce, results summary.
- Clean and comment code; make sure a fresh clone runs (`pip install -r requirements.txt`, then the commands in the README).
- Review the paper against the PDF checklist; fill the contributions table honestly (each student's actual tasks).
- Push final commit **before the deadline**; no changes afterwards.

## Task split (4 students)
| Student | Main ownership |
|---|---|
| A | Repo/Git, dataset, EDA, preprocessing/augmentation, README |
| B | Models, freeze policies, training loop, CNN from scratch |
| C | Experiments run (regimes, CV, hyperparameters), results CSV |
| D | Evaluation, ROC/AUC, Grad-CAM, figures |
All: paper sections (each writes the sections matching their code), cross-review, related-work references.

## Verification (how we know it is correct)
- Unit sanity: split has no overlap between train/val/test (assert on file hashes); class counts per split printed.
- Each model: forward pass shape check; trainable-param count matches the freeze policy (extract ≪ partial < full).
- Overfit test: a model reaches ~100% train accuracy on a 32-image subset.
- Re-run one experiment with same seed gives same metrics (reproducibility).
- Test-set metrics computed once, after model selection on validation.
- Paper numbers = numbers in `results/*.csv`; each figure comes from a script.
- Final check against the PDF checklist (all sections, English, GitHub link in abstract, contributions table, 4 required models + scratch CNN, 3 regimes, Grad-CAM, CV).

## Open items to confirm after download
Exact dataset class names/counts, whether images are grayscale/duplicated, whether the chosen EfficientNetV2 variant (S) is acceptable, and the final deadline date (the PDF only says "exam day").
