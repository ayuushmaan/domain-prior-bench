"""Track-A pretraining (Week 4): one run = one (alpha, seed, size).

Live-sampled prior (infinite unique tasks, seeded workers) -> fixed-size
batches -> nanoTabPFN fork. Checkpoints + JSONL log + dev-only eval.
Eval on TEST sets is forbidden here (see bench/run.py in Week 7).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from domainprior.models.nanotfm import build_model, count_params, prep_task, NanoTFMClassifier, N_TR, N_TE
from domainprior.priors.mixture import sample_mixture


class LivePrior(Dataset):
    def __init__(self, alpha, run_seed, total, n_tr, n_te, max_cols):
        self.alpha, self.run_seed, self.total = alpha, run_seed, total
        self.n_tr, self.n_te, self.max_cols = n_tr, n_te, max_cols

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.run_seed * 1_000_003 + idx * 7_777) % 2**31)
        t = sample_mixture(rng, self.alpha)
        Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng,
                                       max_cols=self.max_cols, n_tr=self.n_tr, n_te=self.n_te)
        x = np.vstack([Xtr, Xte]).astype(np.float32)
        y = np.concatenate([ytr, yte]).astype(np.int64)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y).long()}


def pad_collate(batch):
    """Pad feature dim to batch max with zeros (= constant column; the model's
    per-column norm + clip makes it near-ignorable). Rows/labels fixed by prep."""
    dmax = max(b["x"].shape[1] for b in batch)
    xs = []
    for b in batch:
        x = b["x"]
        if x.shape[1] < dmax:
            pad = torch.zeros(x.shape[0], dmax - x.shape[1], dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        xs.append(x)
    return {"x": torch.stack(xs), "y": torch.stack([b["y"] for b in batch])}


def worker_init(wid):
    info = torch.utils.data.get_worker_info()
    np.random.seed((info.dataset.run_seed + wid * 10_007) % 2**31)


def dev_eval(model, device, seed=0, eval_ctx=100, eval_test=200):
    from domainprior.bench.datasets import DEV_KEYS, load_dataset
    from domainprior.bench.metrics import compute_metrics
    from sklearn.model_selection import StratifiedShuffleSplit
    clf = NanoTFMClassifier(model, device)
    out = {}
    for key in DEV_KEYS:
        try:
            X, y, _ = load_dataset(key, n_cap=3000, seed=seed)
        except Exception as e:
            out[key] = {"error": str(e)}
            continue
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y).astype(int)
        lls, aucs = [], []
        sss = StratifiedShuffleSplit(n_splits=2, test_size=eval_test, random_state=seed)
        for tr, te in sss.split(Xa, ya):
            sel = np.random.default_rng(seed).choice(tr, min(eval_ctx, len(tr)), replace=False)
            try:
                p = clf.predict_proba(Xa[sel], ya[sel], Xa[te])[:, 1]
                m = compute_metrics(ya[te], p)
                lls.append(m["log_loss"]); aucs.append(m["roc_auc"])
            except Exception:
                continue
        if lls:
            out[key] = {"log_loss": round(float(np.mean(lls)), 4),
                        "roc_auc": round(float(np.nanmean(aucs)), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="SMALL")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ntr", type=int, default=100)
    ap.add_argument("--nte", type=int, default=50)
    ap.add_argument("--max-cols", type=int, default=32)
    ap.add_argument("--eval-ctx", type=int, default=100)
    ap.add_argument("--eval-test", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/trackA")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    run = f"prior{str(a.alpha).replace('.', 'p')}_seed{a.seed}_{a.size}_{a.ntr}x{a.nte}"
    outdir = Path(a.out) / run
    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / "train.jsonl", "a")

    def emit(rec):
        rec.update(run=run, alpha=a.alpha, seed=a.seed, size=a.size, t=time.time())
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{run}] " + " ".join(f"{k}={v}" for k, v in rec.items()
                                     if k in ("step", "loss", "steps_s", "dev", "ckpt")))

    torch.manual_seed(a.seed)
    model = build_model(a.size, a.seed).to(device)
    emit({"event": "start", "params": count_params(model), "device": str(device)})
    import schedulefree
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=a.lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()
    ds = LivePrior(a.alpha, a.seed, a.steps * a.accum * a.batch, a.ntr, a.nte, a.max_cols)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    worker_init_fn=worker_init, collate_fn=pad_collate, drop_last=True)
    model.train(); opt.train()
    opt.zero_grad()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    t0 = time.time(); done = 0
    for step, b in enumerate(dl, 1):
        x, y = b["x"].to(device), b["y"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
            out = model((x, y[:, :a.ntr].float()), train_test_split_index=a.ntr)
            loss = crit(out.reshape(-1, out.shape[-1]), y[:, a.ntr:].reshape(-1).long()) / a.accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % a.accum == 0:
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad()
            done += 1
            dt = time.time() - t0
            emit({"step": done, "loss": round(float(loss.item() * a.accum), 4),
                  "steps_s": round(done / dt, 2)})
            if done % (a.eval_every // 1) == 0 or done == a.steps:
                model.eval(); opt.eval()
                emit({"step": done, "dev": dev_eval(model, device, a.seed,
                                                    a.eval_ctx, a.eval_test),
                      "ckpt": str(done)})
                torch.save(model.state_dict(), outdir / f"ckpt_{done}.pt")
                model.train(); opt.train()
            if done >= a.steps:
                break
    torch.save(model.state_dict(), outdir / "final.pt")
    emit({"event": "done", "wall_h": round((time.time() - t0) / 3600, 2)})
    log.close()


if __name__ == "__main__":
    main()
