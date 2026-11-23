# Personal plan: Houcine El Kabaoui

Role: training engine, freeze policies, CPU benchmark, hyperparameter study, ResNeXt-101, EfficientNetV2. Shared overview: `PLAN.md`. Interfaces: `CONTRACT.md`.

## Code I own
`src/train.py`, `src/models/__init__.py`, `src/models/base.py`, `src/models/resnext.py`, `src/models/efficientnet.py`, `scripts/benchmark_cpu.py`, `tests/test_models.py`, `results/tuning/`.

## Phase 0 (days 1-2)
- [ ] Python 3.11/3.12 venv, `pip install -r requirements.txt`, clone the repo
- [ ] Smoke test ResNeXt-101 and EfficientNetV2 on dummy data: create, forward pass, parameter count, time of 1 batch forward+backward
- [ ] `scripts/benchmark_cpu.py`: timing table for all models (also the other students' models), shared with the team. This sets the final `batch_size`, `epochs`, `cv.folds` and `subset_fraction` in `configs/base.yaml`
- [ ] Read `CONTRACT.md` with the others (1 hour) and confirm the laptop specs (CPU, RAM)

## Phase 2: models and training code (days 3-7)
- `base.py`: `apply_regime` (extract, partial, full) with a test that trainable parameters satisfy extract << partial < full
- `resnext.py` (`resnext101_32x8d`) and `efficientnet.py` (`tf_efficientnetv2_s`, name the exact variant in the paper)
- `train.py`: AdamW with a different LR for head and backbone, cosine schedule, early stopping on validation loss, per-epoch history, wall-clock time, checkpoint every epoch with resume, writes the run folder described in `CONTRACT.md`
- Overfit test on 32 images; same seed gives the same metrics

## Phase 3: experiments (days 7-14)
- ResNeXt-101 and EfficientNetV2 in 3 regimes under the common protocol; CV for extract and partial regimes
- Small hyperparameter study (LR, unfreeze depth) on one or two models in `results/tuning/`
- If a full fine-tuning epoch takes more than about 1 hour, use a stratified subset and document it

## Paper (days 10-20)
II Related work (10-15 real papers, each verified from the source), III.5-III.6 (models, training parameters), IV.1-IV.2 (hardware/software, learning curves). I include the model paragraphs of the other two.

## Phase 6 (days 19-21)
Clean and comment my code, cross-review the paper.
