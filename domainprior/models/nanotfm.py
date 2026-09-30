"""Track-A fork of nanoTabPFN (automl/nanoTabPFN, Apache-2.0; vendored in
third_party/nanoTabPFN, unmodified). This module adds what the paper's
educational code omits and our prior needs:

- size configs (SMALL ~2M for Week 4; BASE/LARGE for the Week-5 ladder)
- NaN-safe + categorical-code-tolerant preprocessing (median impute on train
  stats, missingness dummies, column/row caps for 6GB VRAM)
- context-capped sklearn-style classifier for dev evaluation
- exact task batching (fixed R_tr/R_te/D per batch, stratified row subsample)

Only the prior changes between Track-A runs; everything here is fixed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "nanoTabPFN"))
from model import NanoTabPFNModel, NanoTabPFNClassifier  # noqa: E402

SIZES = {
    # emb, heads, mlp, layers  (param counts verified in test)
    "SMALL": dict(embedding_size=160, num_attention_heads=5, mlp_hidden_size=640,
                  num_layers=6, num_outputs=2),
    "BASE": dict(embedding_size=256, num_attention_heads=8, mlp_hidden_size=1024,
                 num_layers=8, num_outputs=2),
    "LARGE": dict(embedding_size=384, num_attention_heads=8, mlp_hidden_size=1536,
                  num_layers=10, num_outputs=2),
}

MAX_COLS = 64
N_TR, N_TE = 384, 128  # fixed context/query rows per batch (6GB guard)


def build_model(size="SMALL", seed=0, qk_scale=1.0):
    """QK scale sharpens initial attention (Phase-6 finding): default init is
    near-uniform over context rows on synthetic tasks and never bootstraps.
    qk_scale=3 unsticks single-task overfit; keep at 1 for ablations."""
    torch.manual_seed(seed)
    m = NanoTabPFNModel(**SIZES[size])
    if qk_scale != 1.0:
        E = SIZES[size]["embedding_size"]
        with torch.no_grad():
            for b in m.transformer_blocks:
                for attn in (b.self_attention_between_datapoints,
                             b.self_attention_between_features):
                    attn.in_proj_weight.data[:2 * E].mul_(qk_scale)
    return m


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def prep_task(X_tr, y_tr, X_te, y_te, rng, max_cols=MAX_COLS, n_tr=N_TR, n_te=N_TE):
    """numpy in -> fixed-size float32 out. Stratified row subsample (keeps both
    classes), median-impute from train, missing dummies, column cap.
    Returns X_tr, y_tr, X_te, y_te aligned."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr = np.asarray(y_tr).astype(int)
    y_te = np.asarray(y_te).astype(int)

    def strat_sub(X, y, n):
        if len(y) <= n:
            return X, y
        out_x, out_y = [], []
        per = max(1, n // 2)
        for c in (0, 1):
            idx = np.where(y == c)[0]
            take = min(len(idx), per if c == 0 else n - per)
            sel = rng.choice(idx, take, replace=False)
            out_x.append(X[sel]); out_y.append(np.full(take, c))
        rest = n - sum(map(len, out_y))
        if rest > 0:
            extra = rng.choice(len(y), rest, replace=False)
            out_x.append(X[extra]); out_y.append(y[extra])
        return np.vstack(out_x), np.concatenate(out_y)

    X_tr, y_tr = strat_sub(X_tr, y_tr, n_tr)
    if len(X_te) > n_te:  # test: plain subsample (may be single-class; eval skips those)
        sel = rng.choice(len(X_te), n_te, replace=False)
        X_te, y_te = X_te[sel], y_te[sel]
    # bootstrap-pad short tasks to exact shapes (batching needs fixed R)
    if len(y_tr) < n_tr:
        extra = rng.choice(len(y_tr), n_tr - len(y_tr), replace=True)
        X_tr, y_tr = np.vstack([X_tr, X_tr[extra]]), np.concatenate([y_tr, y_tr[extra]])
    if len(y_te) < n_te:
        extra = rng.choice(len(y_te), n_te - len(y_te), replace=True)
        X_te, y_te = np.vstack([X_te, X_te[extra]]), np.concatenate([y_te, y_te[extra]])
    assert len(y_tr) == n_tr and len(y_te) == n_te
    d = X_tr.shape[1]
    if d > max_cols // 2:  # leave headroom for missing dummies under MAX_COLS
        keep = rng.choice(d, max_cols // 2, replace=False)
        X_tr, X_te = X_tr[:, keep], X_te[:, keep]
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    has_miss = np.isnan(X_tr).any(axis=0) | np.isnan(X_te).any(axis=0)
    M_tr = np.isnan(X_tr).astype(np.float64)
    M_te = np.isnan(X_te).astype(np.float64)
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    if has_miss.any():
        X_tr = np.concatenate([X_tr, M_tr[:, has_miss]], axis=1)
        X_te = np.concatenate([X_te, M_te[:, has_miss]], axis=1)
    if X_tr.shape[1] > max_cols:
        X_tr, X_te = X_tr[:, :max_cols], X_te[:, :max_cols]
    return (X_tr.astype(np.float32), y_tr.astype(np.int64),
            X_te.astype(np.float32), y_te.astype(np.int64))


class NanoTFMClassifier:
    """Eval-only wrapper: capped context, NaN-safe, chunked test inference."""

    def __init__(self, model, device, ctx_cap=512):
        self.model = model.to(device).eval()
        self.device = device
        self.ctx_cap = ctx_cap

    @torch.no_grad()
    def predict_proba(self, X_train, y_train, X_test):
        from domainprior.models.wrappers import _prep
        import pandas as pd
        Xt = pd.DataFrame(np.asarray(X_train)); Xe = pd.DataFrame(np.asarray(X_test))
        Xtr, Xte = _prep(Xt, Xe)
        ytr = np.asarray(y_train).astype(int)
        if len(Xtr) > self.ctx_cap:
            sel = np.random.default_rng(0).choice(len(Xtr), self.ctx_cap, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        x = np.vstack([Xtr, Xte]).astype(np.float32)
        outs = []
        for s in range(0, len(Xte), 256):
            e = min(len(Xte), s + 256)
            xx = torch.from_numpy(np.vstack([Xtr, Xte[s:e]])).unsqueeze(0).to(self.device)
            yy = torch.from_numpy(ytr).unsqueeze(0).float().to(self.device)
            out = self.model((xx, yy), train_test_split_index=len(Xtr)).squeeze(0)
            outs.append(F.softmax(out[:, :2], dim=1).cpu().numpy())
        return np.vstack(outs)
