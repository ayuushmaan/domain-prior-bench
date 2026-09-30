"""Generate -> sharded HDF5 task dumps. Each worker writes its own shard.

Layout per shard dumps/tasks_s{shard}.h5:
  /t{i:06d}/X_train float32 (NaN = missing), y_train int8, X_test, y_test,
           is_cat uint8[d], families vlen-utf8[d]; group attrs = Task.meta
  root attrs: config JSON, seed, alpha, generator version.

Manifest: dumps/manifest.json (shards, counts, config hash).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from domainprior.priors.mixture import sample_mixture

GENERATOR_VERSION = "credit-v1"
DOM_KW = ("n_rows", "n_feat", "pos_rate", "approval", "p_oot", "p_drift",
          "use_mnar", "use_rules", "use_selection", "use_heaping",
          "use_outliers", "p_outlier", "use_blocks")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _write_shard(job: dict) -> dict:
    import h5py  # imported in worker (Windows spawn-safe)
    rng = np.random.default_rng(job["seed"])
    path = Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    dom = {k: v for k, v in job.items() if k in DOM_KW}
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(job["config"], default=str)
        f.attrs["seed"] = job["seed"]
        f.attrs["alpha"] = job["alpha"]
        f.attrs["generator"] = GENERATOR_VERSION
        for i in range(job["n_tasks"]):
            t = sample_mixture(rng, job["alpha"], **dom)
            g = f.create_group(f"t{i:06d}")
            g.create_dataset("X_train", data=np.asarray(t.X_train, dtype=np.float32),
                             compression="lzf")
            g.create_dataset("y_train", data=np.asarray(t.y_train, dtype=np.int8))
            g.create_dataset("X_test", data=np.asarray(t.X_test, dtype=np.float32),
                             compression="lzf")
            g.create_dataset("y_test", data=np.asarray(t.y_test, dtype=np.int8))
            g.create_dataset("is_cat", data=np.asarray(t.is_cat, dtype=np.uint8))
            g.create_dataset("families", data=np.asarray(t.families, dtype=str_dt))
            for k, v in t.meta.items():
                g.attrs[k] = str(v) if isinstance(v, bool) else (v if isinstance(v, (int, float, str)) else str(v))
            g.attrs["mixture"] = t.meta.get("mixture", "?")
    return {"shard": job["shard"], "path": str(path), "n_tasks": job["n_tasks"]}


def dump_tasks(outdir="dumps", n_tasks=20000, tasks_per_shard=2000, alpha=0.5,
               seed=0, workers=None, **dom_kw) -> Path:
    outdir = Path(outdir)
    cfg = {"n_tasks": n_tasks, "tasks_per_shard": tasks_per_shard, "alpha": alpha,
           "seed": seed, "generator": GENERATOR_VERSION, **dom_kw}
    h = config_hash(cfg)
    outdir = outdir / f"a{alpha}_seed{seed}_{h}"
    n_shards = (n_tasks + tasks_per_shard - 1) // tasks_per_shard
    jobs = []
    for s in range(n_shards):
        n = min(tasks_per_shard, n_tasks - s * tasks_per_shard)
        jobs.append({"shard": s, "path": str(outdir / f"tasks_s{s:04d}.h5"),
                     "n_tasks": n, "seed": seed * 10_000 + s,
                     "alpha": alpha, "config": cfg, **dom_kw})
    workers = workers or min(os.cpu_count() or 4, n_shards)
    done = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_write_shard, jobs):
            done.append(r)
            print(f"shard {r['shard']}: {r['n_tasks']} tasks -> {r['path']}")
    (outdir / "manifest.json").write_text(json.dumps(
        {"config": cfg, "hash": h, "shards": done,
         "n_tasks": sum(d["n_tasks"] for d in done)}, indent=1))
    print(f"manifest -> {outdir / 'manifest.json'}")
    return outdir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dumps")
    ap.add_argument("--n-tasks", type=int, default=20000)
    ap.add_argument("--tasks-per-shard", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    dump_tasks(a.out, a.n_tasks, a.tasks_per_shard, a.alpha, a.seed, a.workers)


if __name__ == "__main__":
    main()
