"""Aggregate results/raw/*.json -> table with mean + 95% bootstrap CI, ranks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["log_loss", "roc_auc", "ece", "brier"]


def load_raw(raw_dir):
    rows = []
    for p in Path(raw_dir).glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("status") != "ok":
            rows.append({"model": r.get("model"), "dataset": r.get("dataset"),
                         "size": r.get("size"), "status": r.get("status"),
                         "metric": None, "value": None})
            continue
        for m in METRICS:
            rows.append({"model": r["model"], "dataset": r["dataset"], "size": r["size"],
                         "status": "ok", "metric": m, "value": r["metrics"][m]})
    return pd.DataFrame(rows)


def bootstrap_ci(x, n_boot=1000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(raw_dir, out_csv):
    df = load_raw(raw_dir)
    ok = df[df.status == "ok"]
    agg = []
    for (model, ds, size, metric), g in ok.groupby(["model", "dataset", "size", "metric"]):
        mean, lo, hi = bootstrap_ci(g["value"].to_numpy())
        agg.append({"model": model, "dataset": ds, "size": size, "metric": metric,
                    "mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "n": len(g)})
    agg = pd.DataFrame(agg)
    # mean rank per (dataset,size,metric): lower log_loss/ece/brier better, higher auc better
    if len(agg):
        parts = []
        for (ds, size, metric), g in agg.groupby(["dataset", "size", "metric"]):
            asc = metric in ("log_loss", "ece", "brier")
            g = g.copy()
            g["rank"] = g["mean"].rank(ascending=asc, method="min")
            # normalized score 0=worst 1=best per (dataset,size,metric)
            mn, mx = g["mean"].min(), g["mean"].max()
            if mx - mn < 1e-12:
                g["norm"] = 0.5
            elif asc:
                g["norm"] = (mx - g["mean"]) / (mx - mn)
            else:
                g["norm"] = (g["mean"] - mn) / (mx - mn)
            parts.append(g)
        agg = pd.concat(parts, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(agg)} rows)")
    # console headline: log_loss means per size
    sub = agg[agg.metric == "log_loss"].pivot_table(index="model", columns="size", values="mean")
    print(sub.to_string())
    # report skips/fails
    bad = df[df.status != "ok"]
    if len(bad):
        print("\nskipped/failed:")
        print(bad.groupby(["model", "status"]).size().to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/week1_table.csv")
    a = ap.parse_args(argv)
    aggregate(a.raw, a.out)


if __name__ == "__main__":
    main()
