"""Coverage: PCA of synthetic meta-feature cloud + real dev overlay.

Reports per-real-point percentile distance to synthetic centroid and
whether each real point falls inside the synthetic PC min-max box.
Saves CSV + PNG (results/validation/).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def coverage(synth_mf: pd.DataFrame, real_mf: pd.DataFrame, labels=None) -> dict:
    cols = synth_mf.columns.tolist()
    scaler = StandardScaler().fit(synth_mf.fillna(0).to_numpy(float))
    Zs = scaler.transform(synth_mf.fillna(0).to_numpy(float))
    Zr = scaler.transform(real_mf.fillna(0).to_numpy(float))
    pca = PCA(n_components=2).fit(Zs)
    Ps, Pr = pca.transform(Zs), pca.transform(Zr)
    cen = Ps.mean(0)
    ds = np.linalg.norm(Ps - cen, axis=1)
    out = {"pca_var": [round(float(v), 3) for v in pca.explained_variance_ratio_],
           "points": []}
    labels = labels or [f"real_{i}" for i in range(len(real_mf))]
    lo, hi = Ps.min(0), Ps.max(0)
    for i, (name, p) in enumerate(zip(labels, Pr)):
        dr = float(np.linalg.norm(p - cen))
        pct = float((ds <= dr).mean())  # 1.0 = farther than all synthetic
        inside = bool((p >= lo).all() and (p <= hi).all())
        out["points"].append({"dataset": name, "pc1": round(float(p[0]), 3),
                              "pc2": round(float(p[1]), 3),
                              "dist_pct": round(pct, 3), "inside_box": inside})
    out["n_inside"] = sum(p["inside_box"] for p in out["points"])
    return {"result": out, "synth_pc": Ps, "real_pc": Pr,
            "synth_cols": cols}


def save_plot(synth_pc, real_pc, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(synth_pc[:, 0], synth_pc[:, 1], s=4, alpha=0.3, label="synthetic")
    ax.scatter(real_pc[:, 0], real_pc[:, 1], s=60, marker="x", label="real dev")
    for (x, y), name in zip(real_pc, labels):
        ax.annotate(name, (x, y), fontsize=8)
    ax.set(xlabel="PC1", ylabel="PC2", title="Prior coverage: synthetic cloud + real dev")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
