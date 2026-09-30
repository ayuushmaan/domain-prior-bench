"""Phase 7 (CPU): qk_scale sweep x robust-encoder on the standard hard task.
6 configs x 300 steps. Log to results/phase7_cpu.log. No GPU.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from torch import nn
from domainprior.priors.credit import sample_credit_task
from domainprior.models.nanotfm import build_model, prep_task
from sklearn.metrics import roc_auc_score
import sys as _s
_s.path.insert(0, "third_party/nanoTabPFN")
import model as _vendored

DEVICE = torch.device("cpu")
_orig_fwd = _vendored.FeatureEncoder.forward


def robust_forward(self, x, train_test_split_index):
    x = x.unsqueeze(-1)
    mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
    std = torch.clamp(torch.std(x[:, :train_test_split_index], dim=1, keepdims=True), min=1e-2)
    xn = torch.nan_to_num((x - mean) / std, nan=0.0, posinf=5.0, neginf=-5.0)
    return self.linear_layer(torch.clip(xn, min=-5.0, max=5.0))


def run(name, qk, robust, steps=300):
    _vendored.FeatureEncoder.forward = robust_forward if robust else _orig_fwd
    rng = np.random.default_rng(0)
    t = sample_credit_task(np.random.default_rng(42), n_rows=(300, 300))
    Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng,
                                   max_cols=32, n_tr=100, n_te=50)
    m = build_model("SMALL", 0, qk_scale=qk).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
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


for qk, rob in [(1.0, False), (2.0, False), (3.0, False), (5.0, False),
                (3.0, True), (5.0, True)]:
    run("qk%s_rob%s" % (qk, int(rob)), qk, rob)
print("PHASE7 DONE", flush=True)
