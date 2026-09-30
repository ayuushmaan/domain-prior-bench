"""Text-interpreted probe of synthetic prior tasks (Q2 aid).

Renders each sampled Task as a human-readable text table (family-named
columns, '?' for MNAR missing, header with OOT/drift/rules meta) and scores
HGB vs NanoTFM side-by-side. Big gaps = places the TFM misreads the table.

CPU-only. Writes results/text_table.csv + prints meta-sliced gap summary.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from domainprior.priors.mixture import sample_mixture
from domainprior.models.nanotfm import build_model, prep_task, NanoTFMClassifier


def render_task(t, max_cols=8, n_train=8, n_test=5) -> str:
    names, counts = [], {}
    for f in t.families:
        counts[f] = counts.get(f, 0) + 1
        names.append("%s_%d" % (f, counts[f]))
    d = min(max_cols, t.X_train.shape[1])

    def fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "?"
        return ("%.1f" % v) if isinstance(v, float) else str(v)

    L = ["# task mixture=%s oot=%s drifted=%s rules=%d k=%d approval=%.2f pos=%.2f miss=%.3f" % (
        t.meta.get("mixture"), t.meta.get("oot"), t.meta.get("drifted"),
        t.meta.get("n_rules"), t.meta.get("k"), t.meta.get("approval"),
        t.meta.get("pos_rate_train"), t.meta.get("missing_rate"))]
    L.append(" | ".join(names[:d] + ["label"]) + "   [TRAIN]")
    for i in range(min(n_train, len(t.y_train))):
        L.append(" | ".join([fmt(t.X_train[i, j]) for j in range(d)] + [str(int(t.y_train[i]))]))
    L.append(" | ".join(names[:d] + ["true"]) + "   [TEST]")
    for i in range(min(n_test, len(t.y_test))):
        L.append(" | ".join([fmt(t.X_test[i, j]) for j in range(d)] + [str(int(t.y_test[i]))]))
    return "\n".join(L)


def score_task(t, clf_rand, clf_trained, rng, ctx=100, nte=50):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    Xtr, ytr, Xte, yte = prep_task(
        t.X_train, t.y_train, t.X_test, t.y_test, rng,
        max_cols=32, n_tr=ctx, n_te=nte)
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        return None
    out = {}
    h = HistGradientBoostingClassifier(max_iter=100).fit(Xtr, ytr)
    ph = h.predict_proba(Xte)[:, 1]
    out["hgb_auc"] = round(float(roc_auc_score(yte, ph)), 3)
    out["hgb_ll"] = round(float(log_loss(yte, ph)), 3)
    for name, clf in (("rand", clf_rand), ("tfm", clf_trained)):
        try:
            p = clf.predict_proba(Xtr, ytr, Xte)[:, 1]
            out[name + "_auc"] = round(float(roc_auc_score(yte, p)), 3)
            out[name + "_ll"] = round(float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))), 3)
        except Exception as e:
            out[name + "_auc"], out[name + "_ll"] = None, None
            out[name + "_err"] = repr(e)[:80]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--ctx", type=int, default=100)
    ap.add_argument("--ckpt", default="results/trackA/prior0p0_seed0_SMALL/final.pt")
    ap.add_argument("--out", default="results/text_table.csv")
    ap.add_argument("--no-mnar", action="store_true")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--no-outliers", action="store_true")
    a = ap.parse_args(argv)
    dom_kw = {"use_mnar": not a.no_mnar, "use_rules": not a.no_rules,
              "use_blocks": not a.no_blocks, "use_outliers": not a.no_outliers}

    dev = torch.device("cpu")
    m_rand = build_model("SMALL", 123)
    m_tr = build_model("SMALL", 123)
    ckpt = Path(a.ckpt)
    if ckpt.exists():
        m_tr.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print("loaded trained ckpt: %s" % ckpt)
    else:
        print("WARN: ckpt missing (%s), trained arm = second random init" % ckpt)
    clf_rand = NanoTFMClassifier(m_rand, dev, ctx_cap=a.ctx)
    clf_trained = NanoTFMClassifier(m_tr, dev, ctx_cap=a.ctx)

    rows = []
    for i in range(a.n_tasks):
        rng = np.random.default_rng(a.seed + i)
        t = sample_mixture(np.random.default_rng(a.seed * 1000 + i), a.alpha, **dom_kw)
        s = score_task(t, clf_rand, clf_trained, rng, ctx=a.ctx)
        if s is None:
            continue
        row = {"task": i, "mixture": t.meta.get("mixture"), "oot": t.meta.get("oot"),
               "drifted": t.meta.get("drifted"), "n_rules": t.meta.get("n_rules"),
               "k": t.meta.get("k"), "miss": round(t.meta.get("missing_rate"), 4),
               "pos": round(t.meta.get("pos_rate_train"), 3), **s}
        row["gap_hgb_tfm"] = (round(row["hgb_auc"] - (row["tfm_auc"] or 0), 3)
                              if row["hgb_auc"] is not None and row["tfm_auc"] is not None else None)
        rows.append(row)
        if i == 0:
            print(render_task(t))
            print("-" * 60)
        print("task %d %s oot=%s rules=%d hgb=%.3f tfm=%.3f gap=%.3f" % (
            i, row["mixture"], row["oot"], row["n_rules"],
            row["hgb_auc"] or -1, row["tfm_auc"] if row["tfm_auc"] is not None else -1,
            row["gap_hgb_tfm"] if row["gap_hgb_tfm"] is not None else -9))

    out = Path(a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gaps = np.array([r["gap_hgb_tfm"] for r in rows if r["gap_hgb_tfm"] is not None])
    print("wrote %s (%d tasks), mean HGB-TFM gap=%.3f max=%.3f" % (
        out, len(rows), gaps.mean(), gaps.max()))
    for key in ("oot", "drifted"):
        for v in (False, True):
            g = [r["gap_hgb_tfm"] for r in rows if r[key] == v and r["gap_hgb_tfm"] is not None]
            if g:
                print("  %s=%s: n=%d mean_gap=%.3f" % (key, v, len(g), float(np.mean(g))))


if __name__ == "__main__":
    main()
