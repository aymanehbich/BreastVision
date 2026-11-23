# CONTRACT: interfaces between the 3 students

Status: **draft v1**, to be reviewed by Aymane, Houcine and Fatima, then tagged `contract-v1`.
After the tag it changes only with the agreement of the 3 students (open a pull request that edits this file).

Everything below is checked by `tests/` and exercised on dummy data first (`data.dummy: true` in `configs/base.yaml`).

## 1. Shared protocol
`configs/base.yaml` is the single source of hyperparameters, identical for all models and regimes.
Nobody hard-codes seed, batch size, epochs, learning rates or paths. Load it with `src.common.load_config()`.
Values marked *provisional* are fixed after the CPU benchmark (Houcine).

## 2. Data (owner: Aymane)
```python
from src.data import get_loaders
train, val, test = get_loaders(cfg, fold=None)   # torch DataLoaders
```
- Batches are `(images, labels)`: images `float32 [B, 3, 224, 224]` (grayscale replicated to 3 channels, ImageNet normalisation), labels `int64 [B]`, **0 = benign, 1 = malignant**.
- `fold=None`: fixed split from the manifest. `fold=k` (0..cv.folds-1): cross-validation on train+val, val = fold k. The test loader is identical in both cases.
- `data.dummy: true` returns random images with balanced labels, so nobody waits for the dataset.
- Source of truth for splits: `results/manifest.csv` (committed), one row per full mammogram. Columns used by everyone: `image_id, patient_id, label, split` (`train|val|test`), `fold` (`0..k-1` on train+val, `-1` on test). Other columns are metadata (view, side, density, ...).
  The plan mentioned `data/splits.csv`; `results/manifest.csv` replaces it (it lives in Git, `data/` does not).
- Splits are **by patient**. `check_no_leakage` asserts that no patient is in two splits.
- Images must be cached once: `python -m src.data cache` (writes `data/cache/224/`, git-ignored).

## 3. Models (one file per model in `src/models/`)
Each model file exposes exactly:
```python
def build(num_classes: int, timm_name: str) -> torch.nn.Module     # pretrained ImageNet weights, new head
def head_params(model) -> list[torch.nn.Parameter]                  # head only, trained in every regime
def unfreeze_partial(model) -> None                                 # unfreeze the last stage (regime "partial")
def gradcam_target_layers(model) -> list[torch.nn.Module]           # for pytorch-grad-cam
```
MaxViT additionally exposes `reshape_transform(tensor)`.
`timm_name` comes from `cfg["models"][name]` (`scratch_cnn` ignores it and has random init).

| Config name | File | Owner |
|---|---|---|
| `xception` | `xception.py` | Fatima |
| `resnext101` | `resnext.py` | Houcine |
| `efficientnetv2` | `efficientnet.py` | Houcine |
| `maxvit` | `maxvit.py` | Aymane |
| `scratch_cnn` | `scratch_cnn.py` | Aymane |

`src/models/base.py` (Houcine) applies the regime: `apply_regime(model, module, mode)` with
`extract` (head only), `partial` (head + last stage), `full` (all). Expected trainable parameters: extract << partial < full.
Use `from src.models import build_model, get_module` to create a model by config name.

## 4. Training (owner: Houcine)
```python
from src.train import run
run_dir = run(model, mode, cfg, fold=None, final=False)   # mode: "extract" | "partial" | "full"
```
Early stopping on validation loss, checkpoint every epoch with resume (checkpoints stay local in `checkpoints/`, not in Git).
`final=True` is the only call allowed to read the test loader.

## 5. Run folder
`results/runs/<model>_<mode>_f<fold>/` (`f` + fold number, or `fnone`), created by `src.common.run_dir(model, mode, fold)`:

| File | Content |
|---|---|
| `history.csv` | per epoch: `epoch, train_loss, train_acc, val_loss, val_acc, epoch_time_s` |
| `val_preds.csv` | `image_id, label, prob_malignant` on the validation images |
| `meta.json` | model, mode, fold, total params, trainable params, training time, hardware (`src.common.hardware_info()`), torch version, git commit, config used |
| `test_preds.csv` | same columns as `val_preds.csv`, **only when `final=True`** |

Each owner writes only the run folders of their own models, so there are no Git conflicts.
Evaluation (`src/evaluate.py`, `scripts/aggregate_results.py`) reads these files and nothing else.

## 6. Test set rule
The test set is used **once**, at the end, after model selection on validation (and CV), with `final=True`.

## 7. Git rules
- One branch per person (`aymane/...`, `houcine/...`, `fatima/...`), small pull requests into `main`, commit at each step.
- Every file has one owner (`.github/CODEOWNERS`). To change a file you do not own, ask its owner.
- `data/`, `checkpoints/`, caches and virtual environments are never committed.
