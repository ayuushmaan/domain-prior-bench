"""Credit SCM prior (§4): latent traits -> feature families -> monotone risk.

Each mechanism has a switch for Track-A ablations:
  use_mnar, use_rules, use_selection, use_heaping (+p_oot/p_drift probs).
Returns a Task (train/test split, OOT or random). Degenerate resamples.
"""
from __future__ import annotations
import numpy as np
from .base import Task

FAMILIES = ["income", "ratio", "count", "tenure", "score", "category", "noise"]
FAM_P = [0.15, 0.20, 0.20, 0.10, 0.10, 0.15, 0.10]
MNAR_FAMS = {"score", "count", "tenure"}  # bureau-type fields


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_credit_task(rng, n_rows=(128, 2048), n_feat=(6, 80), pos_rate=(0.002, 0.60),
                       approval=(0.4, 1.0), p_oot=0.3, p_drift=0.4,
                       use_mnar=True, use_rules=True, use_selection=True, use_heaping=True,
                       use_outliers=True, p_outlier=0.02, use_blocks=True,
                       _depth=0) -> Task:
    if _depth > 8:
        raise RuntimeError("too many degenerate resamples; widen pos_rate?")
    n = int(rng.integers(n_rows[0], n_rows[1] + 1))
    d = int(rng.integers(n_feat[0], n_feat[1] + 1))
    k = int(rng.integers(1, 4))
    appr = float(rng.uniform(*approval)) if use_selection else 1.0
    m = int(np.ceil(n / appr))  # applicant pool before approval

    t = np.sort(rng.uniform(0, 1, m))          # application time
    Z = rng.standard_normal((m, k))            # latent traits
    drifted = bool(rng.random() < p_drift)
    if drifted:
        Z[:, 0] += rng.normal(0, 0.8) * t      # macro shock on main trait

    fams = rng.choice(FAMILIES, size=d, p=FAM_P)
    X = np.empty((m, d)); is_cat = np.zeros(d, bool)
    # block loadings (default): each feature reads mostly ONE latent trait,
    # so features correlate in blocks (payment history / balances / tenure).
    # Dense loadings (ablation) smear every trait into every feature.
    blocks = rng.integers(0, k, size=d) if (use_blocks and k > 1) else None
    for j, fam in enumerate(fams):
        if fam == "noise":
            w = np.zeros(k)
        elif blocks is not None and rng.random() < 0.8:
            w = np.zeros(k)  # hard block: one trait drives this feature
            w[blocks[j]] = rng.uniform(0.7, 1.8) * rng.choice([-1.0, 1.0])
        else:
            w = rng.normal(0, 1, k)
        if fam != "noise":
            w[0] += abs(rng.normal(0.7, 0.5))  # dominant capacity factor: shared background correlation
        s = Z @ w + rng.normal(0, rng.uniform(0.1, 0.6), m)
        if fam == "income":
            x = np.exp(rng.normal(10, 0.5) + rng.uniform(0.3, 0.9) * s)
            if use_heaping:
                x = np.round(x, -int(rng.integers(2, 4)))
        elif fam == "ratio":
            x = np.clip(_sig(s) * rng.uniform(0.6, 1.5), 0, None)
        elif fam == "count":
            lam = np.exp(rng.normal(-1, 0.7) - 0.8 * s)
            x = rng.poisson(lam) * (rng.random(m) > rng.uniform(0.3, 0.8))  # zero-inflated
        elif fam == "tenure":
            x = np.floor(rng.gamma(2, 1, m) * np.exp(0.4 * s) * 12)  # months
        elif fam == "score":
            x = np.round(300 + 550 * _sig(s))
        elif fam == "category":
            c = int(rng.integers(2, 12))
            edges = np.quantile(s, np.sort(rng.uniform(0, 1, c - 1)))
            x = rng.permutation(c)[np.digitize(s, edges)].astype(float)
            is_cat[j] = True
        else:
            x = rng.standard_normal(m)
        X[:, j] = x

    # accounting-style outliers: rare wild values in numeric fields
    # (dev: polish ratios with kurt >1000). Ablatable via use_outliers.
    if use_outliers:
        num_j = [jj for jj, fam in enumerate(fams) if fam in
                 ("income", "ratio", "count", "tenure", "score")]
        for jj in num_j:
            if rng.random() < 0.7:
                mask = rng.random(m) < p_outlier
                if mask.any():
                    mag = np.exp(rng.uniform(np.log(10), np.log(1000), mask.sum()))
                    X[mask, jj] *= np.where(rng.random(mask.sum()) < 0.8, mag, 1.0)

    # default: monotone in latent risk + optional interaction + lender-rule jumps
    beta = np.abs(rng.normal(1, 0.5, k))
    logit = -(Z @ beta)
    if k > 1:
        logit += rng.normal(0, 0.5) * Z[:, 0] * Z[:, 1]
    n_rules = 0
    if use_rules:
        n_rules = int(rng.integers(0, 3))
        for _ in range(n_rules):
            j = int(rng.integers(d))
            cut = np.quantile(X[:, j], rng.uniform(0.7, 0.95))
            logit += rng.uniform(0.5, 2.0) * (X[:, j] > cut)
    target = float(rng.uniform(*pos_rate))
    lo, hi = -30.0, 30.0
    for _ in range(50):  # bisection: hit target default rate
        b = (lo + hi) / 2
        if _sig(logit - b).mean() > target:
            lo = b
        else:
            hi = b
    y = (rng.random(m) < _sig(logit - b)).astype(np.int64)

    # thin-file applicants miss bureau-type fields (MNAR). Calibrated soft:
    # dev missing is ~0-1%, but test sets (HELOC, GiveMeCredit) are heavy,
    # so keep the mechanism with a heavy tail, median near ~0.5%.
    if use_mnar:
        thin = _sig(-2 * Z[:, -1] + rng.normal(-2, 1))
        for j in np.where(np.isin(fams, ["score", "count", "tenure"]))[0]:
            if rng.random() < 0.35:
                X[rng.random(m) < thin * rng.uniform(0.1, 0.8), j] = np.nan

    # lender approves on noisy view of traits; only approved outcomes observed
    approve_score = Z @ beta + rng.normal(0, rng.uniform(0.3, 1.0), m)
    keep = np.sort(np.argsort(-approve_score)[:n])  # keep time order
    X, y = X[keep], y[keep]

    col = rng.permutation(d)  # column position carries no signal
    X, is_cat, fams = X[:, col], is_cat[col], fams[col]
    n_tr = int(n * rng.uniform(0.5, 0.8))
    oot = bool(rng.random() < p_oot)
    order = np.arange(n) if oot else rng.permutation(n)
    tr, te = order[:n_tr], order[n_tr:]
    if y[tr].min() == y[tr].max():  # degenerate task: resample (keywords: _depth is last)
        return sample_credit_task(rng, n_rows, n_feat, pos_rate, approval, p_oot, p_drift,
                                  use_mnar, use_rules, use_selection, use_heaping,
                                  use_outliers, p_outlier, use_blocks, _depth + 1)
    miss_rate = float(np.isnan(X).mean())
    return Task(X_train=X[tr], y_train=y[tr], X_test=X[te], y_test=y[te],
                is_cat=is_cat, families=[str(f) for f in fams],
                meta={"pos_rate_train": float(y[tr].mean()), "pos_rate_all": float(y.mean()),
                      "missing_rate": miss_rate, "oot": oot, "drifted": drifted,
                      "n_rules": n_rules, "k": k, "approval": appr})
