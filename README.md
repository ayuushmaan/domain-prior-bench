# DomainPrior — credit-risk track (Week 1: harness + baselines)

GPU here: RTX 3050 6 GB (plan assumed 24–48 GB). Week 1 is CPU-first + small-context
TFM inference only. Track B deferred; Track A-small is the scaling target.

## One-command reproduction (Week 1 done-criteria)

```bash
# from TFM2.0/, using the TFM venv (has tabicl/torch/sklearn) :
$env:PYTHONPATH = (Get-Location).Path
..\TFM\.venv\Scripts\python.exe scripts\run_week1.py --config configs\week1_baselines.yaml
..\TFM\.venv\Scripts\python.exe -m domainprior.bench.aggregate --raw results\raw --out results\week1_table.csv
```

Done when: `results/week1_table.csv` exists with rows for every
(model, dataset, size) in the config + 95% bootstrap CIs, and
`splits/*.parquet` hashes verify.

## Layout

- `domainprior/bench/` — datasets, splits, stress, run, metrics, aggregate
- `domainprior/models/wrappers.py` — one sklearn-style interface for every baseline;
  missing optional deps → job status `skipped`, never a crash
- `domainprior/priors/` — stubbed until Weeks 2–3 (credit SCM lives in `credit.py`)
- `configs/week1_baselines.yaml` — the experiment grid
- `splits/` — committed frozen split indices (hash-verified)
- `results/raw/` — committed per-job JSON (including failures/skips)

## Baselines (Week 1)

| family | key | needs |
|---|---|---|
| WoE scorecard | `woe_logreg` | sklearn only |
| GBDT | `hgb` (sklearn HistGradientBoosting, always runs), `lightgbm`, `catboost`, `xgboost` | optional |
| Deep tabular | `realmlp` (sklearn MLP stub) | optional torch later |
| Foundation | `tabiclv2`, `tabpfn`, `limix`, `tabdpt`, `mitra` | tabicl runs; rest skip gracefully until installed |
| Ceiling | `autogluon` | skipped until installed |

## Data

Dev (may influence prior): australian_credit, credit_approval, polish_bankruptcy.
Test (frozen, run once at end — Week 1 smoke uses tiny subsamples only):
german_credit, taiwan_default, heloс, give_me_credit, home_credit, lendingclub,
ulb_fraud, ieee_fraud. Large sets are subsampled; see `datasets.py` licences.

## Weeks 2–3 status (prior + validation, 2026-09-23)

## Week 4 status (Track A — on box sandbox sb-5d0c56f4824874e7, 2026-09-24)

- Regime: SMALL ~2.58M params, 100×50 tasks, batch 32, fp16, schedulefree,
  2500 steps/run. 5 priors (α 0/0.25/0.5/0.75/1) × 3 seeds = 15 runs.
- Smoke on box: 5.3 steps/s, 16GB peak VRAM, loss drops.
- Box cells: prior (azMf), validate (CdKT), model (ESah), smoke (BfeO/spJJ),
  sweep (WNZd, background thread → /tmp/trackA/*.jsonl).
- Run 1 (generic s0) done ~10min: dev AUC ~0.5 (chance, as on local).
  Full sweep ETA ~2.5h from 08:25 UTC. Old sandbox (f1868101) expired (HTTP 410).
- SWEEP RESULT (box #2 also expired 2026-09-24, HTTP 410; 14/15 final evals
  captured, domain-s2 lost at step 900): ALL runs chance-level AUC (0.36–0.60).
  Train loss falls with alpha (0.57 generic → 0.52 domain) but no transfer.
  Polish log-loss improves with alpha (0.35→0.30) at flat AUC = calibration
  of base rate only, not ranking. Q1 at this budget: NEGATIVE.
  Weights lost with sandbox (no ckpts saved) — disambiguator needs box #3.

- Local 512-ctx pretraining killed after diagnosis: 2×2500-step generic runs
  learned the prior loss (0.77→0.61) but in-prior AUC stayed 0.52 = chance.
  Regime too big to learn in-budget → rescoped to paper-scale 100×50 tasks
  (batch 32). See `scripts/diag_transfer.py`.
- Fixed real bug: degenerate-resample passed `_depth` positionally into
  `use_outliers` (now keywords) + regression test.
- Marimo box paired: prior + validation ports as notebook cells, box-side
  §5 check reproduces local numbers (PPC 0.68 vs local 0.70).
  Box ports: `C:\Users\ayush\AppData\Local\Temp\opencode\box_prior.py`,
  `box_validate.py`. Box has CUDA (torch 2.11+cu130) — training there is possible.

- `priors/credit.py` — full §4 SCM with all switches + `use_outliers`, `use_blocks`,
  dominant capacity factor; calibrated defaults (see `configs/prior_calibration.md`)
- `priors/generic.py` — lightweight placeholder until nanoTabPFN dumps/priorforge wire-in
- `priors/mixture.py` — α-mixture; `train/dump.py` — sharded HDF5 dumps (seed+config
  per shard, manifest); `train/loader.py` — round-trip + padded collate for Track A
- `diagnostics/` + `scripts/validate_prior.py` — coverage PCA, C2ST, PPC on dev only:
  C2ST 0.99, coverage 2/3 dev inside box, PPC gap −0.26 (synth harder — conservative)
- 11 tests pass (`tests/test_prior.py` + `tests/test_harness.py`)
- Production dump DONE: `dumps/a0.5_seed0_3f1c3aba8d15/` — 20,000 tasks,
  10 shards, manifest-verified (disk was freed via pip-cache purge, 0.03→14.6GB).

## Close-out status (2026-09-29) + future work

Done and committed: Week-1 harness, validated prior (v4), dump, full Q1
sweep 15/15 (`results/trackA_summary_final.csv`: 6 local + 9 molab),
text-table probe + ablations (`scripts/text_table.py`), reader-collapse
diagnosis (`configs/collapse_findings.md`: uniform attention ->
constant readout -> dead grads; QK scaling delays but does not cure).

Parked for later (needs pretrained backbone, not the from-scratch fork):
Q3 (domain-synthetic CPT of TabICLv2 vs TabPFN-3.5/LimiX-2, TabArena guard),
Q4 (synthetic vs real in-domain CPT), locked 8-test-set final eval (dev-only
until then — test sets untouched). First step when resumed: fix or replace
the reader (in-prior AUC gate >0.7), then re-run Q1.
