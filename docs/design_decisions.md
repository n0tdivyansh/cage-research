# Design decisions (resolving the open questions)

## OQ1 / Q3 — Representing collectives
**Decision:** expand each collective into the point-to-point transfers of the algorithm that ran
(`cage/collectives.py`), keep the op name on every event, and record the assumed algorithm in
`trace.meta['collective_algo']`.
**Why:** the network carries the algorithm's transfers, not the semantic intent; standard pairwise graph
algorithms (Louvain, METIS) apply directly; the op name lets a different expansion be swapped in later.
**Consequence:** when ranks move, libraries like NCCL rebuild ring order. Collective traffic expanded in
rank order is therefore not "pinned" to ranks. `placement.pinned_matrix` separates rank-pinned traffic
(all-to-all, p2p, pipeline, halo) from re-routable collectives; every placement result reports both
regimes (`fixed`, `pinned`). Hypergraph view: `collectives.participants()`, unused by algorithms.

## OQ2 / Q1 — Static vs. per-step graphs
**Decision:** the stored object is an event table with a step index (`CommTrace`). The static aggregate,
per-step matrices and any window are views (`matrix(steps=...)`, `step_matrices()`, `subset()`).
**Why:** I-GAPE needs sequences; Paper 1 needs aggregates; one data model serves both with no rewrite.
Steps come from `MPI_Pcontrol(1)` (PMPI shim), `ProfilerStep#N` (Chakra), or `ops_per_step` (NCCL logs).

## OQ3 / Q2 — Fidelity vs. overhead
**Decision:** process (rank) granularity; record bytes, message count and, where available, duration.
Sampling every k-th step is supported (`subset(steps=range(0, n, k))`). E6 measures how much fidelity is
lost; I1 shows when sampling breaks (non-stationary routing).

## OQ4 — Ground truth
**Decision:** all three options, in order: (A) planted partitions with known blocks — E1; (B) workloads
whose structure is known by construction (Megatron groups, stencils) — E3/E4; (C) injected anomalies —
E5, with the slow-NIC injection as the real test and volume injection as a sanity check.
Real-trace validation (B with NAS/HPCG, C on hardware) is open.

## Q4 — Online vs. offline
**Decision:** Paper 1 is offline analysis of a fixed observation window. Online, adaptive behaviour is the
subject of Papers 4–5 (I-GAPE triggers, GAPE engine), simulated with explicit migration cost.

## Modularity must be tested
Every reported Q comes with a degree-preserving null model (double-edge swaps carrying weights; weight
shuffling for dense graphs) and z / empirical p. Reports call structure "significant" only with
≥ 19 null samples, z > 3 and p ≤ 0.05.

## Placement baseline and capacity
Baseline = block placement. Partitions are exact-capacity (8 ranks per host): METIS (`ufactor=1`) →
greedy capacity repair → best-improvement pairwise swap refinement. `method="best"` takes the lowest cut of
METIS-seeded, block-seeded and random-seeded refinement, so the recommendation is never worse than block.
Recommendations are renamed (Hungarian matching) to minimise the number of ranks that move.

## Cost model
Bandwidth-bound per-host estimate (`placement.py`, `gape/topology.py`). It ranks placements; it does not
predict seconds. Results from it are scenario values (CLAUDE.md rule E1).
