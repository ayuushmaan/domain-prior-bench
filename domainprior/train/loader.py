"""Read back dumped tasks; batch collate with NaN-mask + padding (for Track A)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from domainprior.priors.base import Task


def iter_tasks(shard_path):
    import h5py
    with h5py.File(shard_path, "r") as f:
        for name in sorted(f.keys()):
            g = f[name]
            fam = [s.decode() if isinstance(s, bytes) else str(s) for s in g["families"][:]]
            meta = {k: v for k, v in g.attrs.items()}
            yield Task(X_train=g["X_train"][:], y_train=g["y_train"][:].astype(int),
                       X_test=g["X_test"][:], y_test=g["y_test"][:].astype(int),
                       is_cat=g["is_cat"][:].astype(bool), families=fam, meta=meta)


def count_tasks(shard_path) -> int:
    import h5py
    with h5py.File(shard_path, "r") as f:
        return len(f.keys())


def collate(tasks: list[Task]):
    """Pad a batch to (B, N_max, D_max). Returns dict with *_mask (1 = real)."""
    B = len(tasks)
    N = max(max(len(t.y_train), len(t.y_test)) for t in tasks)
    D = max(t.X_train.shape[1] for t in tasks)
    Xtr = np.full((B, N, D), np.nan, np.float32)
    Xte = np.full((B, N, D), np.nan, np.float32)
    ytr = np.full((B, N), -1, np.int64)
    yte = np.full((B, N), -1, np.int64)
    mtr = np.zeros((B, N), np.float32)
    mte = np.zeros((B, N), np.float32)
    for b, t in enumerate(tasks):
        n1, n2, d = len(t.y_train), len(t.y_test), t.X_train.shape[1]
        Xtr[b, :n1, :d] = t.X_train
        Xte[b, :n2, :d] = t.X_test
        ytr[b, :n1] = t.y_train
        yte[b, :n2] = t.y_test
        mtr[b, :n1] = 1.0
        mte[b, :n2] = 1.0
    return {"X_train": Xtr, "y_train": ytr, "train_mask": mtr,
            "X_test": Xte, "y_test": yte, "test_mask": mte}
