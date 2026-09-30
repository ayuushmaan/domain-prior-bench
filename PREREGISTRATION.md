# Preregistration (fill date before first test run; do not edit after)

Date: 2026-09-23. Test datasets are NOT to be used for any design decision
until Week 7 freeze. Dev-only tuning until then.

## Q1. Domain prior > generic prior at equal compute?
H: mixture prior improves log-loss on held-out credit sets, same arch/steps.
Refute: no significant difference (Wilcoxon over 8 test sets, Holm-corrected).

## Q2. Where do gains come from?
H: gains concentrate in MNAR missingness / heavy imbalance / out-of-time splits.
Refute: uniform gains, or per-mechanism ablations change nothing.

## Q3. Domain-synthetic CPT upgrades open SOTA?
H: continued pretraining of TabICLv2 on the mixture narrows gap to
TabPFN-3.5/LimiX-2 on credit data with <1% normalized-score loss on TabArena subset.
Refute: no in-domain gain or general collapse.

## Q4. Synthetic as substitute for real in-domain data?
H: CPT on domain-synthetic matches CPT on real dev sets.
Refute: real data wins clearly (itself publishable).

Checkpoint/alpha selection: dev sets only. Preprocessing fit inside each split.
Every run (incl. failures/timeouts) committed as raw JSON.
