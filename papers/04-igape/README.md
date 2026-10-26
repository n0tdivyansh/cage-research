# Paper 4 — I-GAPE: iteration-aware analysis and re-planning

**Question.** How stable is the communication graph across iterations, which similarity measure
detects placement-relevant change, and how often should a placement be recomputed?

**Data model.** No new extraction: `CommTrace` keeps per-step events (design decision OQ2).

## Findings (I1, scenario values; MoE with 4 routing regimes of 30 steps)
- Rank-aligned relative-L1 drift spikes 20× at regime boundaries; spectral distance only 3.2× because it
  is permutation-invariant (blind to *which* ranks talk). The roadmap's planned "spectral / edit
  distance" triggers are the wrong tool for placement.
- Total comm time: block 202 s, plan-once 166 s, periodic 116 s, triggered 112 s, oracle 102 s.
- Triggered re-planning reaches within 10% of the oracle with 4 re-plans (threshold 0.3).

## Open tasks
- Real multi-iteration traces: does routing drift at all in real MoE training (load-balancing losses
  suppress it)? Curriculum / data-mixture changes are the likely source.
- Gradual drift (not just abrupt regimes); hysteresis on the trigger.
- Structural fingerprints for *reusing* a previous plan when an old regime returns.
