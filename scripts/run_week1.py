"""One-command Week-1 reproduction: grid from YAML -> raw JSON jobs."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
from domainprior.bench.run import run_job


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/week1_baselines.yaml")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    jobs = [(m, d, s, r) for m in cfg["models"] for d in cfg["datasets"]
            for s in cfg["sizes"] for r in range(cfg.get("n_repeats", 2))]
    print(f"{len(jobs)} jobs: {len(cfg['models'])} models x {len(cfg['datasets'])} "
          f"datasets x {len(cfg['sizes'])} sizes x {cfg.get('n_repeats', 2)} repeats")
    for m, d, s, r in tqdm(jobs):
        out = run_job(m, d, s, r, seed=cfg.get("seed", 0),
                      n_repeats=cfg.get("n_repeats", 2),
                      outdir=cfg.get("outdir", "results/raw"),
                      n_cap=cfg.get("n_cap", 6000),
                      subsample_test=cfg.get("subsample_test", 500))
    print("done. Now run: python -m domainprior.bench.aggregate --raw results/raw --out results/week1_table.csv")


if __name__ == "__main__":
    main()
