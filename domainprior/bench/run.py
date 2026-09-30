"""One (model, dataset, split, size) job -> results/raw/*.json. Never crashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np

from domainprior.bench import datasets as D
from domainprior.bench import splits as S
from domainprior.bench.metrics import compute_metrics
from domainprior.models.wrappers import fit_predict


def job_id(model, dataset, size, repeat, seed):
    h = hashlib.sha256(f"{model}|{dataset}|{size}|{repeat}|{seed}".encode()).hexdigest()[:10]
    return f"{model}__{dataset}__n{size}__r{repeat}_{h}"


def run_job(model, dataset, size, repeat, seed=0, n_repeats=10, outdir="results/raw",
            n_cap=20000, subsample_test=2000):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    jid = job_id(model, dataset, size, repeat, seed)
    out = outdir / f"{jid}.json"
    if out.exists():
        return out
    t0 = time.time()
    rec = {"job_id": jid, "model": model, "dataset": dataset, "size": size,
           "repeat": repeat, "seed": seed, "status": "ok"}
    try:
        X, y, tm = D.load_dataset(dataset, n_cap=max(n_cap, size * 2), seed=seed)
        splits = S.make_splits(y, n_repeats=n_repeats, seed=seed,
                               time=(tm if tm is not None else None), oot=False)
        sp = splits[repeat]
        # subsample train pool to `size` (stratified where possible)
        tr_idx = np.array(sp["train_idx"]); te_idx = np.array(sp["test_idx"])
        ytr_pool = np.asarray(y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx])
        if len(tr_idx) > size:
            rng = np.random.default_rng(seed + repeat)
            pos = tr_idx[ytr_pool == 1]; neg = tr_idx[ytr_pool == 0]
            n_pos = min(len(pos), max(1, int(size * ytr_pool.mean())))
            sel = np.concatenate([rng.choice(pos, n_pos, replace=False) if len(pos) else [],
                                  rng.choice(neg, size - n_pos, replace=False)])
            tr_idx = np.sort(sel.astype(int))
        if len(te_idx) > subsample_test:  # keep TFM inference cheap on 6GB
            rng = np.random.default_rng(seed + 999)
            te_idx = np.sort(rng.choice(te_idx, subsample_test, replace=False))
        Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        ytr = (y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx])
        Xte = X.iloc[te_idx] if hasattr(X, "iloc") else X[te_idx]
        yte = np.asarray(y.iloc[te_idx] if hasattr(y, "iloc") else y[te_idx]).astype(int)
        f0 = time.time()
        p = fit_predict(model, Xtr, ytr, Xte, seed=seed + repeat)
        rec["fit_predict_s"] = round(time.time() - f0, 2)
        rec["metrics"] = compute_metrics(yte, p)
        rec["n_train"], rec["n_test"] = int(len(Xtr)), int(len(Xte))
        rec["base_rate"] = float(np.mean(yte))
    except ImportError as e:
        rec.update(status="skipped", reason=f"missing dep: {e}")
    except Exception as e:
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc(limit=5))
    rec["wall_s"] = round(time.time() - t0, 2)
    out.write_text(json.dumps(rec, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--outdir", default="results/raw")
    a = ap.parse_args(argv)
    print(run_job(a.model, a.dataset, a.size, a.repeat, a.seed, a.n_repeats, a.outdir))


if __name__ == "__main__":
    main()
