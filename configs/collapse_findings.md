# Reader-collapse findings (SMALL/BASE nanoTabPFN fork, Sep 2026)

Q1 swept clean (15/15, loss falls with alpha, dev AUC chance everywhere).
Before concluding "domain prior fails", we checked whether the reader can
learn at all. It cannot. Chain of evidence, cheapest-first:

## Phases (single fixed task unless noted; HGB reference AUC 0.774 there)

| phase | test | result |
|---|---|---|
| overfit | 1 task x 1000 steps, schedulefree+fp16 | loss frozen 0.527, AUC noise 0.33-0.67 |
| gradcheck | per-param grad norms at step 1 | 0/116 dead — grads flow at init |
| Adam+fp32 | same task, plain Adam, no autocast | identical stall 0.5269 — not optimizer/precision |
| robust-enc | floor std 1e-2, clip +-5 (encoder std was 6.3e4) | same wall — blowup real but not causal |
| clean-regime | breast_cancer fixed task | AUC 0.95 from step 1 — arch works on clean real data |
| clean8 | synthetic d=8, no MNAR/outliers | stalls 0.641 — synthetic family itself, not noise |
| QK sweep | init QK x1/x2/x3/x5 on hard task | 300-step AUC 0.35/0.59/0.68/0.46 (x5 spikes 0.70 then collapses) |
| long | QK x3, 2000 steps | decays to chance — delays, does not cure |

## Mechanism (measured, not speculated)

- Phase 3: logit-difference std collapses ~1000x in 20 steps (3.1e-2 -> 8e-6);
  corr with labels +0.14 -> noise. Common-mode output stays healthy.
- Phase 4: block-0 datapoint attention entropy 4.58/4.605 = uniform; unchanged
  after 60 steps. Every query averages all context rows identically.
- Structural trap: decoder reads only the target slot; query target slots are
  identically mean-padded, so with uniform attention all queries are identical
  by construction. Discrimination REQUIRES non-uniform attention, whose
  gradients are ~100x weaker than the common path at init — the constant
  predictor wins first, loss plateaus, grads vanish (med 1e-2 -> 1e-12).

## Verdict

Q1-negative is a capacity/optimization verdict, not a prior verdict: the reader
never learned to in-context-learn (in-prior AUC 0.51/0.499 even at BASE).
Q2-probe gaps (HGB 0.64 vs TFM 0.51, uniform across ablations) are consistent:
uniform reader failure, no mechanistic concentration.
Do not run more from-scratch sweeps. Next: pretrained-backbone CPT (Q3/Q4).
