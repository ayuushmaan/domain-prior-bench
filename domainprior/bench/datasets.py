"""Dataset registry + loaders. Dev may shape the prior; test is frozen.

Small sets load from OpenML/sklearn into data/ cache. Large sets
(home_credit, lendingclub, ieee_fraud, ulb_fraud, give_me_credit) are
subsampled and documented with licence notes — verify on download.
Falls back to a synthetic credit-like frame when offline so Week-1
smoke tests never block on network.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    role: str          # dev | test
    openml_id: int | None
    target: str
    time_col: str | None
    licence: str
    note: str

REGISTRY: dict[str, DatasetSpec] = {
    # IDs verified against OpenML 2026-09-23 (names/shapes checked, not from memory)
    "australian_credit": DatasetSpec("australian_credit", "dev", 40981, "target", None,
        "CC-BY (verify)", "Statlog Australian, 690x15 incl target"),
    "credit_approval": DatasetSpec("credit_approval", "dev", 46377, "target", None,
        "CC-BY (verify)", "UCI approval, 690x16 incl target, has missing"),
    "polish_bankruptcy": DatasetSpec("polish_bankruptcy", "dev", 42880, "target", "year",
        "CC-BY (verify)", "Corporate 1-year slice, 7027x65, imbalanced"),
    "german_credit": DatasetSpec("german_credit", "test", 31, "target", None,
        "CC-BY (verify)", "Classic small-data case, 1000 rows"),
    "taiwan_default": DatasetSpec("taiwan_default", "test", 42477, "target", None,
        "verify", "Payment-history features, 30k rows"),
    "heloc": DatasetSpec("heloc", "test", 45023, "target", None,
        "verify", "Bureau-style, 10k rows"),
    "give_me_credit": DatasetSpec("give_me_credit", "test", None, "target", None,
        "Kaggle competition (verify)", "150k imbalanced; manual download"),
    "home_credit": DatasetSpec("home_credit", "test", None, "target", None,
        "Kaggle competition (verify)", "307k x 120+; manual download"),
    "lendingclub": DatasetSpec("lendingclub", "test", None, "target", "issue_d",
        "Kaggle mirror removed (verify)", "Out-of-time via issue dates; manual"),
    "ulb_fraud": DatasetSpec("ulb_fraud", "test", 1597, "target", "Time",
        "verify", "Extreme imbalance 0.17%, 284k rows"),
    "ieee_fraud": DatasetSpec("ieee_fraud", "test", None, "target", "TransactionDT",
        "Kaggle competition (verify)", "Wide/messy; manual download"),
}

DEV_KEYS = [k for k, s in REGISTRY.items() if s.role == "dev"]
TEST_KEYS = [k for k, s in REGISTRY.items() if s.role == "test"]


def _synthetic_fallback(key, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10, 0.5, n)
    debt_ratio = np.clip(rng.normal(0.5, 0.2, n), 0, 1.5)
    delinq = rng.poisson(0.4, n)
    bureau = rng.normal(650, 80, n)
    logit = -2.0 + 1.5 * debt_ratio + 0.8 * (delinq > 0) - 0.005 * (bureau - 650)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"income": income, "debt_ratio": debt_ratio,
                      "delinq_24m": delinq, "bureau": bureau})
    return X, pd.Series(y, name="target")


def load_dataset(key, n_cap=20000, seed=0):
    """Returns (X: DataFrame, y: Series, time: Series|None). Offline-safe."""
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}")
    spec = REGISTRY[key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{key}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        y = df[spec.target]
        t = df[spec.time_col] if spec.time_col and spec.time_col in df else None
        return df.drop(columns=[spec.target] + ([spec.time_col] if t is not None else [])), y, t
    # try OpenML for small sets
    if spec.openml_id is not None:
        try:
            from sklearn.datasets import fetch_openml
            bunch = fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto")
            # NOTE: bunch.data = features only; bunch.frame includes target -> never use frame as X
            tgt = bunch.target
            if tgt is None:
                df = bunch.frame.copy()
                tgt = df.iloc[:, -1]
                X_raw = df.iloc[:, :-1].copy()
            else:
                X_raw = bunch.data.copy() if hasattr(bunch.data, "copy") else pd.DataFrame(bunch.data)
                tgt = pd.Series(np.asarray(tgt).ravel())
            _t = pd.Series(tgt).reset_index(drop=True)
            _s = _t.astype(str)
            uniq = sorted(_s.dropna().unique().tolist())
            _tnum = pd.to_numeric(_t, errors="coerce")
            if len(uniq) <= 2:
                # binary target (incl. numeric-coded 0/1): sorted-last = positive.
                # Never median-split: that destroys real imbalance (e.g. fraud).
                pos = uniq[-1]
                y = (_s == pos).astype(int)
            elif _tnum.notna().mean() > 0.9:  # continuous score -> median split
                y = (_tnum > _tnum.median()).astype(int)
            else:
                raise ValueError(f"{key}: non-binary non-numeric target {uniq[:5]}")
            X = X_raw.reset_index(drop=True).copy()
            # encode categoricals simply; keep numeric where possible
            X = pd.get_dummies(X, drop_first=True, dtype="float32")
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
            y = pd.Series(np.asarray(y, dtype=int), name="target")
            if len(X) > n_cap:
                sel = np.random.default_rng(seed).choice(len(X), n_cap, replace=False)
                X, y = X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)
            df_save = X.copy(); df_save[spec.target] = y.to_numpy()
            df_save.to_parquet(cache, index=False)
            return X, y, None
        except Exception as e:
            print(f"[datasets] OpenML fetch failed for {key}: {e}; using synthetic fallback")
    X, y = _synthetic_fallback(key, n=min(n_cap, 2000), seed=seed)
    return X, y, None


def dataset_hash(key):
    return hashlib.sha256(key.encode()).hexdigest()[:8]
