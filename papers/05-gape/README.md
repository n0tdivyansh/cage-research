# Paper 5 — GAPE: graph-aware placement engine

**Question.** Composing CAGE (placement), T-GAPE (topology), I-GAPE (drift triggers) into one loop, is
adaptive re-placement worth it once migration costs are paid — and what guards against it backfiring?

**Engine** (`gape/engine.py`, simulated): windowed observation → rel-L1 drift trigger → topology-aware
plan → cost gate (migrate only if predicted gain over a horizon exceeds migration cost).
C-GAPE caching is **not yet in the loop** (open task).

## Findings (G1, scenario values)
- Migration ≤ 0.1 s per rank: GAPE total 0.53–0.63× block; plan-once 0.89×.
- Migration ≥ 1 s per rank: the cost gate declines every move (GAPE = block); ungated re-planning costs
  1.27× (1 s) to 4.1× (5 s) block; plan-once 1.10–1.93×.
- The cost gate also helps at zero migration cost (145 s vs 153 s) by rejecting plans built on windows
  that straddle a regime change.
- Topology awareness adds little here (147 vs 145 s) because the 2:1 spine is rarely the bottleneck.

## Open tasks
- Realistic migration cost model: optimizer state size / bandwidth, checkpoint-restart vs live migration.
- Put C-GAPE cache planning in the loop; joint objective.
- A real prototype hook: torchrun / SLURM rank-to-host mapping on restart (elastic training).
