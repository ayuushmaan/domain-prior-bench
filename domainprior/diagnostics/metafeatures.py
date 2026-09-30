"""Per-table meta-features for prior validation (§5). Dev data only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

NAMES = ["n_rows", "n_feat", "pos_rate", "missing_rate", "cat_share", "int_share",
         "skew_med", "kurt_med", "heavy_tail_share", "mean_abs_corr", "top_eig_ratio"]


def _num_frame(X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    num = X.select_dtypes(include=[np.number])
    return num


def metafeatures(X, y) -> dict:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    y = np.asarray(y).astype(int)
    n, d = X.shape
    num = _num_frame(X).copy()
    # subsample rows for expensive stats
    if len(num) > 2000:
        num = num.sample(2000, random_state=0)
    mf = {
        "n_rows": float(n), "n_feat": float(d),
        "pos_rate": float(y.mean()),
        "missing_rate": float(X.isna().mean().mean()),
        "cat_share": float(sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns) / d),
    }
    if num.shape[1]:
        with np.errstate(all="ignore"):
            sk = np.abs(skew(num, nan_policy="omit"))
            ku = kurtosis(num, nan_policy="omit")
        mf["skew_med"] = float(np.nanmedian(sk))
        mf["kurt_med"] = float(np.nanmedian(ku))
        mf["heavy_tail_share"] = float(np.nanmean(sk > 2))
        ints = [(np.abs(num[c].dropna() - np.round(num[c].dropna())) < 1e-9).mean() > 0.9
                for c in num.columns]
        mf["int_share"] = float(np.mean(ints))
        c = num.fillna(num.median()).corr().to_numpy()
        up = c[np.triu_indices(c.shape[0], 1)]
        mf["mean_abs_corr"] = float(np.nanmean(np.abs(up))) if up.size else 0.0
        try:
            ev = np.linalg.eigvalsh(np.nan_to_num(c))
            mf["top_eig_ratio"] = float(ev[-1] / ev.sum()) if ev.sum() > 0 else 0.0
        except np.linalg.LinAlgError:
            mf["top_eig_ratio"] = float("nan")
    else:
        mf.update(skew_med=float("nan"), kurt_med=float("nan"),
                  heavy_tail_share=0.0, int_share=0.0,
                  mean_abs_corr=0.0, top_eig_ratio=float("nan"))
    return {k: mf.get(k, float("nan")) for k in NAMES}


def frame_mf(X, y) -> pd.Series:
    return pd.Series(metafeatures(X, y))
