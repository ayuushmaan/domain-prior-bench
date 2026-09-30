"""Single-task overfit probe: can the reader memorize one fixed task?

Usage: python scripts/overfit_probe.py --qk-scale 3 --steps 300 --device cpu
Pass = loss << 0.55 and AUC >> 0.7 with growing pstd.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from domainprior.priors.credit import sample_credit_task
from domainprior.models.nanotfm import build_model, prep_task


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qk-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="SMALL")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)

    rng = np.random.default_rng(0)
    t = sample_credit_task(np.random.default_rng(42), n_rows=(300, 300))
    Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng,
                                   max_cols=32, n_tr=100, n_te=50)
    m = build_model(a.size, a.seed, qk_scale=a.qk_scale).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()
    xb = torch.from_numpy(np.vstack([Xtr, Xte])).unsqueeze(0).to(dev)
    yb_ctx = torch.from_numpy(ytr).unsqueeze(0).float().to(dev)
    yb_q = torch.from_numpy(yte).unsqueeze(0).long().to(dev)
    m.train()
    from sklearn.metrics import roc_auc_score
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        out = m((xb, yb_ctx), train_test_split_index=len(Xtr))
        loss = crit(out.reshape(-1, out.shape[-1]), yb_q.reshape(-1).long())
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                o = m((xb, yb_ctx), train_test_split_index=len(Xtr)).squeeze(0)[:, :2].float()
                p = torch.softmax(o, 1)[:, 1].cpu().numpy()
            try:
                auc = round(float(roc_auc_score(yte, p)), 3)
            except Exception:
                auc = -1
            print("step=%d loss=%.4f auc=%s pstd=%.2e" % (
                step, float(loss.item()), auc, float(p.std())), flush=True)
    print("OVERFIT_PROBE DONE qk=%.1f" % a.qk_scale, flush=True)


if __name__ == "__main__":
    main()
