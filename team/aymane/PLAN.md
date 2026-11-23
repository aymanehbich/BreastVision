# Personal plan: Aymane Hbich

Role: repo and Git, data pipeline, MaxViT-Base, CNN from scratch, README. Shared overview: `PLAN.md`. Interfaces: `CONTRACT.md`.

## Code I own
`src/common.py`, `src/data.py`, `src/models/maxvit.py`, `src/models/scratch_cnn.py`, `configs/base.yaml`, `scripts/preprocess.py`, `scripts/make_splits.py`, `scripts/check_leakage.py`, `tests/test_data.py`, `notebooks/01_eda.ipynb`, `README.md`.

## Phase 0 (days 1-2)
- [x] Scaffold, requirements, config, manifest of 3,103 mammograms, patient-level split, image cache
- [x] `CONTRACT.md`, `CODEOWNERS`, skeleton of `src/`, dummy mode, CV folds in the manifest
- [x] GitHub repo, CODEOWNERS usernames
- [ ] Check that Houcine and Fatima accepted the invitation, protect `main`
- [x] `scripts/make_splits.py`, `scripts/check_leakage.py`, README how-to-run
- [ ] Team review of `CONTRACT.md`, then tag `contract-v1` (does not wait for the benchmark)
- [ ] Check what GPU the school PC has (`nvidia-smi`); if none, agree with the team how MaxViT-Base is run (Kaggle GPU, or a teammate's GPU)
- [ ] Tag `protocol-v1` once Houcine's benchmark has fixed the values in `configs/base.yaml`

## Phase 1: data (days 2-5)
- Duplicate check (file hashes), EDA notebook (class counts, image sizes, sample grid), decide class-imbalance handling
- Preprocessing: 224 letterbox cache (done), 3-channel replication, ImageNet normalisation; augmentation on train only

## Phase 2: models (days 3-7)
- `maxvit.py` (`maxvit_base_tf_224`, with `reshape_transform` for Grad-CAM) and `scratch_cnn.py` (about 4 conv blocks + BN + dropout, random init)
- Smoke test on dummy data: forward shape, parameter counts for the 3 regimes, overfit 32 images

## Phase 3: experiments (days 7-14)
- Run MaxViT (extract, partial, full) and the scratch CNN under the common protocol; write one run folder each
- CV for extract and partial regimes (with Fatima's `cv.py`)

## Paper (days 10-20)
Sections I Introduction, III.1-III.4 (workflow, dataset, preprocessing, augmentation), VI.2 Limits, plus a paragraph on my models in `paper/sections/models_aymane.tex`.

## Phase 6 (days 19-21)
README with exact reproduction commands, fresh-clone check, final push before the deadline.
