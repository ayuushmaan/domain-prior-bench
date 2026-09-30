# Prior calibration log (dev datasets ONLY — test sets untouched)

Date locked: 2026-09-23. Method: `scripts/validate_prior.py` (§5 checks:
meta-feature coverage, C2ST, PPC) against 3 dev sets
(australian_credit 690×34, credit_approval 690×38, polish_bankruptcy 7027×64;
shapes post-encoding, OpenML IDs verified 2026-09-23 after finding three
wrong IDs from memory: 143/27/826 → 40981/46377/42880).

## Validation trajectory (300 domain + 300 generic tasks, 75 resampled real tables)

| round | change | C2ST dom | PPC gap (synth−real) | coverage |
|---|---|---|---|---|
| v1 | plan defaults | 0.97 | nan (bug) | 2/3, aus 0.92 |
| v2 | pos_rate→(0.002,0.60), n_feat→(6,80), outliers, noise→U(0.1? no, 0.2-1.0) | 1.00 | −0.29 | 2/3, aus 0.95 |
| v3 | resampled C2ST (75 tables), varied sizes, PPC degenerate-safe | 1.00 | −0.24 | 2/3 |
| v4 | dominant capacity factor, hard blocks, MNAR soft, encoded-to-encoded MF | 0.99 | −0.26 | 2/3, appr 0.21 |

## Locked knobs (defaults in `priors/credit.py`)

- `pos_rate=(0.002, 0.60)` — was (0.01, 0.30); dev shows 0.44/0.56 (UCI) and 0.04
  (polish); low end covers fraud (0.0017) for later test.
- `n_feat=(6, 80)` — was (6, 48); polish has 64 raw features. Track A subsamples
  ≤64 columns per task (context cap); log the subsample.
- feature noise `U(0.1, 0.6)` — was U(0.3, 1.5); synth tasks were too hard (PPC −0.29).
- dominant capacity factor `w[0] += |N(0.7,0.5)|` (new) + hard block loadings
  `use_blocks` (new, ablatable): real mean|r| 0.07–0.13 vs prior ~0.05.
- outlier injection `use_outliers` (new, ablatable, p=0.02, ×10–1000):
  polish ratios have kurt>1000; without this, skew/kurt uncovered.
- MNAR soft: family-hit 0.5→0.35, strength U(0.1,0.8): dev missing is ~0–1%,
  but HELOC/GiveMeCredit tests are heavy — keep heavy tail, median ≈0.5%.
- unchanged: `approval=(0.4,1.0)`, `p_oot=0.3`, `p_drift=0.4`, rules/heaping/selection.

## Known gaps (honest, not hidden)

- C2ST AUC 0.99: a linear model still separates synthetic from real tables.
  Biggest drivers (|Cohen's d|): mean_abs_corr −1.27, n_feat +0.88.
  Real bureau/payment blocks correlate tighter than a 1–3 factor model makes.
  Fix deferred: explicit block-covariance features (future work, not more churn now).
- Polish bankruptcy sits outside the synthetic PC box (dist_pct 1.0): extreme-tail
  corporate ratios. Partial coverage accepted; fraud/corporate tails are where
  Q2 expects the domain prior to be tested, not assumed perfect.
- PPC gap −0.26: synthetic tasks are HARDER than curated dev (HGB 0.68 vs 0.94).
  Conservative direction (won't flatter Track A), but watch: if Track A underfits
  clean data, revisit noise downward.
- `cat_share` d=0.0 is partly an encoding artifact (both sides dummified) — fine.
