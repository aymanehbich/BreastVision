"""Smoke tests for the data interface (owner: Aymane). Run: python -m pytest tests  (or python tests/test_data.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config
from src.data import get_loaders


def test_dummy_loaders_follow_the_contract():
    cfg = load_config()
    cfg["data"]["dummy"] = True
    cfg["data"]["num_workers"] = 0
    for fold in (None, 0):
        train, val, test = get_loaders(cfg, fold)
        x, y = next(iter(train))
        s = cfg["data"]["image_size"]
        assert tuple(x.shape[1:]) == (3, s, s) and str(x.dtype) == "torch.float32"
        assert set(y.tolist()) <= {0, 1}


if __name__ == "__main__":
    test_dummy_loaders_follow_the_contract()
    print("ok")
