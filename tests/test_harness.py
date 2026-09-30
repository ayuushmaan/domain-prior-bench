import numpy as np
import pandas as pd
from domainprior.bench.metrics import compute_metrics, ece_score
from domainprior.bench.splits import make_splits, save_splits, verify_splits
from domainprior.models.wrappers import fit_predict


def test_metrics_ranges():
    y = np.array([0, 1, 0, 1] * 25)
    p = np.clip(np.random.default_rng(0).random(100), 0.05, 0.95)
    m = compute_metrics(y, p)
    assert 0 <= m["log_loss"] < 5 and 0 <= m["brier"] <= 1 and 0 <= m["ece"] <= 1
    assert 0 <= m["roc_auc"] <= 1


def test_splits_stratified_and_hash(tmp_path):
    y = pd.Series([0] * 80 + [1] * 20)
    sp = make_splits(y, n_repeats=2, seed=0)
    assert len(sp) == 2 and len(sp[0]["train_idx"]) == 70
    import domainprior.bench.splits as S
    S.SPLIT_DIR = tmp_path
    path, _ = save_splits("dummy", 256, sp)
    assert verify_splits(path)


def test_wrappers_run_offline():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = pd.Series((X["a"] + rng.normal(size=120) > 0).astype(int))
    for key in ("woe_logreg", "hgb", "realmlp"):
        p = fit_predict(key, X.iloc[:80], y.iloc[:80], X.iloc[80:])
        assert len(p) == 40 and np.isfinite(p).all()
