"""Aggregate Track-A sweep: loss curves + dev log-loss vs alpha (Q1 first answer)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

KEYS = ["australian_credit", "credit_approval", "polish_bankruptcy"]


def main():
    root = Path(__file__).resolve().parents[1] / "results" / "trackA"
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "train.jsonl").exists():
            continue
        recs = [json.loads(l) for l in open(d / "train.jsonl")]
        losses = [(r["step"], r["loss"]) for r in recs if "loss" in r]
        devs = [(r["step"], r["dev"]) for r in recs if "dev" in r]
        fin = devs[-1][1] if devs else {}
        rows.append({"run": d.name,
                     "steps": losses[-1][0] if losses else 0,
                     "loss_first50": round(float(np.mean([l for _, l in losses[:50]])), 4) if losses else None,
                     "loss_last100": round(float(np.mean([l for _, l in losses[-100:]])), 4) if losses else None,
                     **{f"{k}_ll": (fin.get(k, {}) or {}).get("log_loss") for k in KEYS},
                     **{f"{k}_auc": (fin.get(k, {}) or {}).get("roc_auc") for k in KEYS}})
    import pandas as pd
    t = pd.DataFrame(rows).sort_values("run")
    t.to_csv(root / "trackA_summary.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
