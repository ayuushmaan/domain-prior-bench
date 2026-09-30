"""Phase 6 (CPU): is the wall task-regime or architecture?
T1 breast_cancer fixed task | T2 clean synthetic d=8 | T3 QKx3 init | T4 LR 1e-2.
Log to results/phase6_cpu.log. No GPU.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from torch import nn
from domainprior.priors.credit import sample_credit_task
from domainprior.models.nanotfm import build_model, prep_task
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cpu")


def get_task(kind):
    rng = np.random.default_rng(0)
    if kind == "breast":
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        X, y = load_breast_cancer(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=50, random_state=0)
        return (np.asarray(Xtr, np.float32), np.asarray(ytr, int),
                np.asarray(Xte, np.float32), np.asarray(yte, int))
    if kind == "clean8":
        r = np.random.default_rng(42)
        t = sample_credit_task(np.random.default_rng(42), n_rows=(300, 300),
                               n_feat=(8, 8), use_mnar=False, use_rules=False,
                               use_outliers=False)
        return prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng,
                         max_cols=32, n_tr=100, n_te=50)
    r = np.random.default_rng(42)
    t = sample_credit_task(r, n_rows=(300, 300))
    return prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng,
                     max_cols=32, n_tr=100, n_te=50)


def run(name, kind, lr=1e-3, qk_scale=1.0, steps=300):
    Xtr, ytr, Xte, yte = get_task(kind)
    m = build_model("SMALL", 0).to(DEVICE)
    if qk_scale != 1.0:
        import sys as _s
        _s.path.insert(0, "third_party/nanoTabPFN")
        for b in m.transformer_blocks:
            for attn in (b.self_attention_between_datapoints,
                         b.self_attention_between_features):
                w = attn.in_proj_weight.data
                e = w.shape[0] // 3
                w[:2 * e] *= qk_scale
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    xb = torch.from_numpy(np.vstack([Xtr, Xte])).unsqueeze(0).to(DEVICE)
    yb_ctx = torch.from_numpy(ytr).unsqueeze(0).float().to(DEVICE)
    yb_q = torch.from_numpy(yte).unsqueeze(0).long().to(DEVICE)
    m.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        out = m((xb, yb_ctx), train_test_split_index=len(Xtr))
        loss = crit(out.reshape(-1, out.shape[-1]), yb_q.reshape(-1).long())
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                o = m((xb, yb_ctx), train_test_split_index=len(Xtr)).squeeze(0)[:, :2].float()
                p = torch.softmax(o, 1)[:, 1].cpu().numpy()
            try:
                auc = round(float(roc_auc_score(yte, p)), 3)
            except Exception:
                auc = -1
            print("%s step=%d loss=%.4f auc=%s pstd=%.2e" % (
                name, step, float(loss.item()), auc, float(p.std())), flush=True)
    print("%s DONE" % name, flush=True)


run("T1-breast", "breast")
run("T2-clean8", "clean8")
run("T3-qkx3", "clean8", qk_scale=3.0)
run("T4-lr1e2", "clean8", lr=1e-2)
print("PHASE6 DONE", flush=True)
