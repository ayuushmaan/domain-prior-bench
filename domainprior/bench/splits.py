"""Frozen stratified splits, hashed. Preprocessing fits inside each split."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SPLIT_DIR = Path(__file__).resolve().parents[2] / "splits"


def _config_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def make_splits(y, n_repeats=10, test_size=0.3, seed=0, time=None, oot=False):
    """Random stratified splits, or out-of-time (train past / test future)."""
    y = np.asarray(y)
    splits = []
    if oot and time is not None:
        order = np.argsort(np.asarray(time))
        n_test = int(len(y) * test_size)
        # same OOT cut repeated with bootstrap resample of train for repeats
        cut = order[:-n_test], order[-n_test:]
        rng = np.random.default_rng(seed)
        for r in range(n_repeats):
            tr = rng.choice(cut[0], size=len(cut[0]), replace=True)
            splits.append({"repeat": r, "train_idx": sorted(map(int, tr)),
                           "test_idx": sorted(map(int, cut[1])), "mode": "oot"})
        return splits
    for r in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + r)
        tr, te = next(sss.split(np.zeros(len(y)), y))
        splits.append({"repeat": r, "train_idx": sorted(map(int, tr)),
                       "test_idx": sorted(map(int, te)), "mode": "random"})
    return splits


def save_splits(dataset, size, splits, seed=0):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset, "size": size, "seed": seed, "splits": splits}
    h = _config_hash({"dataset": dataset, "size": size, "seed": seed,
                      "train": [s["train_idx"][:4] for s in splits]})
    path = SPLIT_DIR / f"{dataset}_n{size}_seed{seed}_{h}.json"
    path.write_text(json.dumps(payload, indent=1))
    return path, h


def verify_splits(path):
    payload = json.loads(Path(path).read_text())
    h = _config_hash({"dataset": payload["dataset"], "size": payload["size"],
                      "seed": payload.get("seed", 0),
                      "train": [s["train_idx"][:4] for s in payload["splits"]]})
    assert Path(path).stem.endswith(h), f"hash mismatch for {path}"
    return True
