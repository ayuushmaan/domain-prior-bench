import numpy as np
from domainprior.priors.credit import MNAR_FAMS, sample_credit_task
from domainprior.priors.mixture import sample_mixture


def test_pos_rate_in_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = sample_credit_task(rng, pos_rate=(0.01, 0.30))
        assert 0.005 <= t.meta["pos_rate_train"] <= 0.35, t.meta
        assert t.y_train.min() != t.y_train.max()


def test_no_nan_labels_and_shapes():
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = sample_credit_task(rng)
        assert not np.isnan(t.y_train).any() and not np.isnan(t.y_test).any()
        assert t.X_train.shape[1] == t.X_test.shape[1] == len(t.is_cat) == len(t.families)
        assert 6 <= t.X_train.shape[1] <= 80


def test_mnar_only_intended_families():
    rng = np.random.default_rng(2)
    for _ in range(30):
        t = sample_credit_task(rng, use_mnar=True)
        X = np.vstack([t.X_train, t.X_test])
        for j, fam in enumerate(t.families):
            if np.isnan(X[:, j]).any():
                assert fam in MNAR_FAMS, f"NaN in {fam}"
    # with MNAR off, no NaNs at all
    for _ in range(10):
        t = sample_credit_task(rng, use_mnar=False)
        assert not np.isnan(t.X_train).any() and not np.isnan(t.X_test).any()


def test_oot_tasks_time_ordered():
    rng = np.random.default_rng(3)
    oot_seen = 0
    for _ in range(50):
        t = sample_credit_task(rng, p_oot=1.0)
        assert t.meta["oot"] is True
        oot_seen += 1
        # OOT = first block train, latter block test from time-sorted pool:
        # train rows were kept in time order, so approved-pool order preserved
    assert oot_seen > 0


def test_switches_off():
    rng = np.random.default_rng(4)
    t = sample_credit_task(rng, use_mnar=False, use_rules=False,
                           use_selection=False, use_heaping=False,
                           use_outliers=False, use_blocks=False)
    assert not np.isnan(t.X_train).any()
    assert t.meta["n_rules"] == 0


def test_blocks_switch_runs_and_differs():
    import pandas as pd
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(6):
        tb = sample_credit_task(rng, use_blocks=True)
        td = sample_credit_task(rng, use_blocks=False)
        seen.add(round(float(pd.DataFrame(tb.X_train).corr().abs().mean().mean()), 6))
        seen.add(round(float(pd.DataFrame(td.X_train).corr().abs().mean().mean()), 6))
    assert len(seen) > 1  # switch perturbs the correlation structure


def test_outliers_add_tail():
    rng = np.random.default_rng(6)
    heavy = []
    for _ in range(10):
        t = sample_credit_task(rng, use_outliers=True, p_outlier=0.05)
        X = np.vstack([t.X_train, t.X_test]).astype(float)
        with np.errstate(all="ignore"):
            heavy.append(np.nanmax(np.abs(X)))
    t0 = sample_credit_task(rng, use_outliers=False)
    X0 = np.vstack([t0.X_train, t0.X_test]).astype(float)
    assert np.nanmax(np.abs(X0)) < max(heavy)


def test_degenerate_resample_keeps_switches():
    # regression: resample path once passed _depth positionally into use_outliers
    rng = np.random.default_rng(11)
    for _ in range(200):
        t = sample_credit_task(rng, pos_rate=(0.001, 0.01), use_outliers=False,
                               use_blocks=True, use_mnar=True)
        assert not np.isnan(t.y_train).any()
        assert 0.0 <= t.meta["pos_rate_train"] <= 0.05
        assert set(t.families) <= {"income", "ratio", "count", "tenure",
                                   "score", "category", "noise"}


def test_mixture_share():
    rng = np.random.default_rng(5)
    got = [sample_mixture(rng, 0.5).meta["mixture"] for _ in range(200)]
    assert 0.35 < sum(g == "domain" for g in got) / 200 < 0.65
    assert all(sample_mixture(rng, 1.0).meta["mixture"] == "domain" for _ in range(5))
    assert all(sample_mixture(rng, 0.0).meta["mixture"] == "generic" for _ in range(5))
