"""Diagnose Track-A transfer: in-prior vs dev AUC for trained vs random model."""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from domainprior.models.nanotfm import build_model, prep_task, NanoTFMClassifier, N_TR
from domainprior.priors.mixture import sample_mixture
from domainprior.train.pretrain import dev_eval

DEVICE = torch.device("cpu")


def inpior_auc(model, alpha, n=12, seed=99):
    rng = np.random.default_rng(seed)
    clf = NanoTFMClassifier(model, DEVICE, ctx_cap=512)
    aucs = []
    from sklearn.metrics import roc_auc_score
    for _ in range(n):
        t = sample_mixture(rng, alpha)
        Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng)
        if len(np.unique(yte)) < 2:
            continue
        p = clf.predict_proba(Xtr, ytr, Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p))
    return round(float(np.mean(aucs)), 3), len(aucs)


for name, ckpt in [("trained-generic-s0", "results/trackA/prior0p0_seed0_SMALL/final.pt"),
                   ("random-init", None)]:
    m = build_model("SMALL", 123)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    ip, n = inpior_auc(m, alpha=0.0)
    print(name, "in-prior(generic) AUC:", ip, f"(n={n})")
    d = dev_eval(m, DEVICE, seed=0)
    print(name, "dev:", {k: v for k, v in d.items()})
