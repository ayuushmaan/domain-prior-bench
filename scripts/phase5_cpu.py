"""Phase 5 (CPU): robust-encoder overfit test. No GPU. Log to results/phase5_cpu.log."""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from torch import nn
from domainprior.priors.credit import sample_credit_task
from domainprior.models.nanotfm import build_model, prep_task
from sklearn.metrics import roc_auc_score
import sys as _sys
_sys.path.insert(0, "third_party/nanoTabPFN")
import model as _vendored


def robust_forward(self, x, train_test_split_index):
    x = x.unsqueeze(-1)
    mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
    std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True)
    std = torch.clamp(std, min=1e-2)
    xn = (x - mean) / std
    xn = torch.nan_to_num(xn, nan=0.0, posinf=5.0, neginf=-5.0)
    xn = torch.clip(xn, min=-5.0, max=5.0)
    return self.linear_layer(xn)


_vendored.FeatureEncoder.forward = robust_forward

rng = np.random.default_rng(0)
t = sample_credit_task(np.random.default_rng(42))
Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng,
                               max_cols=32, n_tr=100, n_te=50)
device = torch.device("cpu")
m = build_model("SMALL", 0).to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
xb = torch.from_numpy(np.vstack([Xtr, Xte])).unsqueeze(0).to(device)
yb_ctx = torch.from_numpy(ytr).unsqueeze(0).float().to(device)
yb_q = torch.from_numpy(yte).unsqueeze(0).long().to(device)
NTR = len(Xtr)
m.train()
for step in range(1, 301):
    opt.zero_grad()
    out = m((xb, yb_ctx), train_test_split_index=NTR)
    loss = crit(out.reshape(-1, out.shape[-1]), yb_q.reshape(-1).long())
    loss.backward()
    opt.step()
    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            o = m((xb, yb_ctx), train_test_split_index=NTR).squeeze(0)[:, :2].float()
            p = torch.softmax(o, 1)[:, 1].cpu().numpy()
        try:
            auc = round(float(roc_auc_score(yte, p)), 3)
        except Exception:
            auc = -1
        print("step=%d loss=%.4f auc=%s pstd=%.4f" % (
            step, float(loss.item()), auc, float(p.std())), flush=True)
with torch.no_grad():
    x = xb.unsqueeze(-1)
    mu = x[:, :NTR].mean(1, keepdims=True)
    sd = x[:, :NTR].std(1, keepdims=True).clamp_min(1e-2)
    xe = m.feature_encoder.linear_layer((((x - mu) / sd).clip(-5, 5)))
    print("enc_rowstd=%.2e" % float(xe.std()), flush=True)
print("PHASE5 DONE", flush=True)
