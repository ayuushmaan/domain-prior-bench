"""Final Q1 aggregate: local completed runs (parsed) + molab 9 (transcribed
from live session table, source-tagged). Writes results/trackA_summary_final.csv.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

KEYS = ["australian_credit", "credit_approval", "polish_bankruptcy"]
ROOT = Path(__file__).resolve().parents[1]

# Transcribed verbatim from molab live agg (2026-09-29, /tmp/tfm/results/*).
MOLAB = {
    # run: (loss_first50, loss_last100, {ds: (ll, auc)})
    "prior0p25_seed0_SMALL_100x50": (0.6779, 0.5756, {"australian_credit": (0.765, 0.467), "credit_approval": (0.880, 0.553), "polish_bankruptcy": (0.344, 0.521)}),
    "prior0p5_seed1_SMALL_100x50": (0.6702, 0.5601, {"australian_credit": (0.778, 0.500), "credit_approval": (0.900, 0.498), "polish_bankruptcy": (0.330, 0.479)}),
    "prior0p5_seed2_SMALL_100x50": (0.6537, 0.5592, {"australian_credit": (0.779, 0.478), "credit_approval": (0.901, 0.441), "polish_bankruptcy": (0.329, 0.507)}),
    "prior0p75_seed0_SMALL_100x50": (0.6303, 0.5426, {"australian_credit": (0.795, 0.509), "credit_approval": (0.926, 0.500), "polish_bankruptcy": (0.313, 0.516)}),
    "prior0p75_seed1_SMALL_100x50": (0.6555, 0.5449, {"australian_credit": (0.768, 0.500), "credit_approval": (0.884, 0.500), "polish_bankruptcy": (0.341, 0.462)}),
    "prior0p75_seed2_SMALL_100x50": (0.6481, 0.5416, {"australian_credit": (0.797, 0.508), "credit_approval": (0.929, 0.506), "polish_bankruptcy": (0.312, 0.486)}),
    "prior1p0_seed0_SMALL_100x50": (0.6168, 0.5303, {"australian_credit": (0.811, 0.546), "credit_approval": (0.949, 0.459), "polish_bankruptcy": (0.300, 0.495)}),
    "prior1p0_seed1_SMALL_100x50": (0.6457, 0.5280, {"australian_credit": (0.809, 0.492), "credit_approval": (0.948, 0.500), "polish_bankruptcy": (0.301, 0.478)}),
    "prior1p0_seed2_SMALL_100x50": (0.6556, 0.5247, {"australian_credit": (0.814, 0.491), "credit_approval": (0.954, 0.506), "polish_bankruptcy": (0.298, 0.392)}),
}


def parse_local(train_path):
    recs = [json.loads(l) for l in open(train_path)]
    losses = [r["loss"] for r in recs if "loss" in r]
    devs = [r["dev"] for r in recs if "dev" in r]
    done = any(r.get("event") == "done" for r in recs)
    fin = devs[-1] if devs else {}
    return done, losses, fin


rows = []
for d in sorted((ROOT / "results" / "trackA").iterdir()):
    tj = d / "train.jsonl"
    if not d.is_dir() or not tj.exists() or "_100x50" not in d.name:
        continue
    if not (d / "final.pt").exists():
        rows.append({"run": d.name, "source": "local", "status": "incomplete-no-final-pt"})
        continue
    done, losses, fin = parse_local(tj)
    r = {"run": d.name, "source": "local",
         "status": "done" if done else "final-pt-no-done-event",
         "steps": len(losses),
         "loss_first50": round(float(np.mean(losses[:50])), 4),
         "loss_last100": round(float(np.mean(losses[-100:])), 4)}
    for k in KEYS:
        m = fin.get(k, {}) or {}
        r[k + "_ll"] = m.get("log_loss")
        r[k + "_auc"] = m.get("roc_auc")
    rows.append(r)

for run, (lf, ll, dev) in MOLAB.items():
    r = {"run": run, "source": "molab-RTX-PRO-6000", "status": "done",
         "steps": 2500, "loss_first50": lf, "loss_last100": ll}
    for k in KEYS:
        r[k + "_ll"], r[k + "_auc"] = dev[k]
    rows.append(r)

out = ROOT / "results" / "trackA_summary_final.csv"
with open(out, "w", newline="") as f:
    fields = ["run", "source", "status", "steps", "loss_first50", "loss_last100"] + \
             [k + s for k in KEYS for s in ("_ll", "_auc")]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["run"]))
print("wrote %s (%d rows)" % (out, len(rows)))
for r in sorted(rows, key=lambda r: r["run"]):
    print(r["run"], r["source"], r["status"])
