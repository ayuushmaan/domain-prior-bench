"""Validate credit prior against DEV datasets only (§5). Test sets untouched.

Outputs results/validation/prior_v1.{json,csv} + coverage.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from domainprior.bench.datasets import DEV_KEYS, load_dataset
from domainprior.diagnostics.metafeatures import frame_mf
from domainprior.diagnostics.c2st import c2st
from domainprior.diagnostics.coverage import coverage, save_plot
from domainprior.diagnostics.ppc import ppc
from domainprior.priors.credit import sample_credit_task
from domainprior.priors.generic import sample_generic_task


def encode_task(t):
    """One-hot encode a synthetic task like the model sees it (_prep parity):
    real dev frames are already dummified, so compare encoded-to-encoded."""
    X = pd.DataFrame(np.asarray(t.X_train))
    for j in np.where(np.asarray(t.is_cat))[0]:
        X[j] = X[j].astype(str)
    X = pd.get_dummies(X, drop_first=True, dtype="float32")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X, t.y_train


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-synth", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/validation")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print("loading dev datasets...")
    dev = {}
    for k in DEV_KEYS:
        try:
            X, y, t = load_dataset(k, n_cap=20000, seed=a.seed)
            dev[k] = (X, y)
            print(f"  {k}: {X.shape}, base_rate={float(np.asarray(y).mean()):.3f}")
        except Exception as e:
            print(f"  {k}: FAILED ({e})")
    real_mf = pd.DataFrame([{**frame_mf(X, y), "dataset": k} for k, (X, y) in dev.items()])

    # resampled real tables: C2ST needs dozens of real points, not 3.
    # 25 stratified bootstrap subsamples per dev set (dev only, descriptive).
    real_tables = []
    for k, (X, y) in dev.items():
        Xa = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
        ya = np.asarray(y).astype(int)
        rng_r = np.random.default_rng(a.seed + abs(hash(k)) % 10_000)
        for _ in range(25):
            n_sub = int(rng_r.integers(200, min(2000, len(ya)) + 1))  # vary size: no n_rows giveaway
        for _ in range(25):
            idx = np.concatenate([rng_r.choice(np.where(ya == c)[0],
                                 size=max(1, int(n_sub * (ya == c).mean())), replace=True)
                                  for c in (0, 1) if (ya == c).any()])
            real_tables.append(frame_mf(Xa.iloc[idx] if hasattr(Xa, "iloc") else Xa[idx],
                                        ya[idx]))
    real_tables_mf = pd.DataFrame(real_tables)

    print(f"sampling {a.n_synth} domain + {a.n_synth} generic tasks...")
    dom = [sample_credit_task(rng) for _ in range(a.n_synth)]
    gen = [sample_generic_task(rng) for _ in range(a.n_synth)]
    dom_mf = pd.DataFrame([frame_mf(*encode_task(t)) for t in dom])
    gen_mf = pd.DataFrame([frame_mf(pd.DataFrame(t.X_train), t.y_train) for t in gen])

    real_feat = real_mf.drop(columns=["dataset"])
    real_tables_feat = real_tables_mf[real_feat.columns]
    res = {
        "n_synth": a.n_synth, "seed": a.seed,
        "c2st_domain": c2st(dom_mf, real_tables_feat),
        "c2st_generic": c2st(gen_mf, real_tables_feat),
        "ppc": ppc(dom[:60], {k: v for k, v in dev.items()}, seed=a.seed),
        "dev_stats": {k: {"n": int(X.shape[0]), "d": int(X.shape[1]),
                          "base_rate": round(float(np.asarray(y).mean()), 4),
                          "missing": round(float(pd.DataFrame(X).isna().mean().mean()), 4)}
                      for k, (X, y) in dev.items()},
    }
    cov = coverage(dom_mf, real_feat, labels=list(dev.keys()))
    res["coverage"] = cov["result"]
    # per-feature standardized gaps (Cohen's d): WHERE does the prior differ?
    gaps = {}
    for c in dom_mf.columns:
        a_ = dom_mf[c].fillna(0).to_numpy(float)
        b_ = real_tables_feat[c].fillna(0).to_numpy(float)
        pooled = np.sqrt((a_.var() + b_.var()) / 2) + 1e-12
        gaps[c] = round(float((a_.mean() - b_.mean()) / pooled), 2)
    res["cohens_d_domain_vs_real"] = gaps
    save_plot(cov["synth_pc"], cov["real_pc"], list(dev.keys()), out / "coverage.png")

    mf_all = pd.concat([dom_mf.assign(prior="domain"),
                        gen_mf.assign(prior="generic"),
                        real_feat.assign(prior="real_dev")], ignore_index=True)
    mf_all.to_csv(out / "prior_v1_metafeatures.csv", index=False)
    (out / "prior_v1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "dev_stats"}, indent=1))
    print(f"wrote {out / 'prior_v1.json'}, metafeatures.csv, coverage.png")


if __name__ == "__main__":
    main()
