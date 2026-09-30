"""Week-4 sweep driver: 5 priors x 3 seeds, sequential on one GPU.

~50 min/run -> ~13h total (~0.5 GPU-day on the 3050). Logs to driver.log.
Resume-safe: skips runs whose final.pt exists.
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
STEPS, BATCH, ACCUM, EVAL_EVERY, WORKERS = 2500, 32, 1, 500, 4
NTR, NTE, MAX_COLS, EVAL_CTX, EVAL_TEST = 100, 50, 32, 100, 200


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "trackA"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "driver.log", "a")
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]
    for i, (a, s) in enumerate(jobs):
        run = f"prior{str(a).replace('.', 'p')}_seed{s}_SMALL_{NTR}x{NTE}"
        if (out / run / "final.pt").exists():
            print(f"[{i + 1}/{len(jobs)}] {run}: done, skipping", flush=True)
            continue
        print(f"[{i + 1}/{len(jobs)}] {run}: starting", flush=True)
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, "domainprior/train/pretrain.py", "--alpha", str(a),
             "--seed", str(s), "--size", "SMALL", "--steps", str(STEPS),
             "--batch", str(BATCH), "--accum", str(ACCUM),
             "--eval-every", str(EVAL_EVERY), "--workers", str(WORKERS),
             "--ntr", str(NTR), "--nte", str(NTE), "--max-cols", str(MAX_COLS),
             "--eval-ctx", str(EVAL_CTX), "--eval-test", str(EVAL_TEST),
             "--out", str(out)],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(f"=== {run} exit={p.returncode} hours={(time.time() - t0) / 3600:.2f} ===\n")
        log.write(p.stdout[-4000:])
        log.flush()
    log.close()
    print("sweep done")


if __name__ == "__main__":
    main()
