# Review findings on the original CAGE documents

Source: stress-test of `docs/original/CAGE_Overview.md` and `docs/original/CAGE_Collaborator_Briefing.md`
(October 2026). Each finding has a status: **fixed in code**, **tested by experiment**, or **open — paper text**.
A reviewer at a systems venue would reject the original framing; these are the reasons, in order of damage.

## 1. Extraction does not match the motivating workload — fixed in code
PyTorch DDP/FSDP on GPUs calls NCCL, not MPI, so PMPI interception sees nothing. Even for MPI codes,
`MPI_Allreduce` is one collective call at the PMPI layer, not the `MPI_Send`/`MPI_Recv` pairs the case
study claims to record.
- Code: `cage/extract/` reads NCCL debug logs and Chakra execution traces for GPU jobs; the PMPI shim
  records collectives as collectives (with communicator membership) and expands them by algorithm.
- Collective representation (OQ1) is now decided: expand by the algorithm that ran. See design_decisions.md.

## 2. Novelty claim is false — open (paper text)
"No existing tool builds a communication graph and applies graph algorithms" is wrong. Prior work to cite
and position against (verify every citation before use):
- Communication-matrix extraction + graph partitioning for rank placement: MPIPP (Chen et al., ICS 2006),
  TreeMatch (Jeannot, Mercier, Tessier), Hoefler & Snir "Generic topology mapping strategies" (ICS 2011),
  LibTopoMap, Scotch-based mapping.
- `MPI_Dist_graph_create(..., reorder=1)` — rank reordering from a declared communication graph is in the MPI standard.
- Vampir / Score-P communication-matrix views.
- ScalAna (TPDS 2025 per earlier notes) — graph-based scaling-loss analysis; closest analog.
- McPherson & Nagarajan (2014/2015) — closest predecessor in approach (per earlier notes).
- ByteScheduler (SOSP 2019), TopoOpt (NSDI 2023) — ML communication scheduling / topology co-design.
Defensible claim: a reproducible pipeline plus an empirical characterisation of *when* communication-graph
structure predicts a placement benefit for modern ML traffic, including negative results.

## 3. Betweenness centrality has no physical meaning here — fixed in code (flagged experimental)
Messages do not route through intermediate ranks, so shortest paths in an application-level graph are not
paths data takes. NetworkX also treats weight as distance, so raw byte weights make heavy edges *long*.
Synchronous jobs die when any rank fails, so the "bridge node fault tolerance" use case does not hold.
- Code: `metrics.betweenness` uses distance = 1/weight and is labelled EXPERIMENTAL; it is off by default
  in reports (`--betweenness` to enable). Recommendation: drop it from the paper or redefine edges as
  real forwarding relationships.

## 4. The 32-worker case study is circular and arithmetically inconsistent — tested by experiment
- Hierarchical allreduce is built around hosts, so Louvain "discovering" host communities is the
  algorithm reading back its own placement. E2/E3 show its modularity is **not significant** against a
  null model (z ≈ 0).
- Weighted-degree sum must equal 2 × traffic: 32 × 387 MB ≈ 12.4 GB is the traffic, so the mean
  degree would be ≈ 775 MB, not 387 MB (`tests/test_core.py::test_strength_sums_to_twice_traffic`).
- "Isolated workers with 38 MB" is a contradiction; every DDP worker takes part in allreduce.
- "Workers 3 and 19 share a data-shard boundary" does not exist in pure data parallelism.
- "Reduce peak load by ~75%" is impossible when the other masters already carry 1.8–1.9 GB.
- The case study contradicts H2: four masters hold ~31% of degree mass, not 70–80%.
Action: rebuild the case study from `cage analyze` output on a real or clearly labelled synthetic trace.

## 5. Hypotheses as written are untestable or wrong
- H1 (Q > 0.3 ⇒ structure): rings, grids and sparse random graphs exceed 0.3 with no structure.
  **Replaced** by Q vs. degree-preserving null model (z, empirical p). E2.
- H2 (80/20 power law): a power law cannot be established on 32–512 nodes; collective-heavy jobs are
  balanced by design. **Replaced** by load concentration (Gini, top-20% share). E3 — expected to fail
  for DP and most HPC, partially hold for skewed MoE.
- H3 (METIS beats round-robin): round-robin is a strawman. **Baseline is now block placement**; NCCL
  rebuilds rings for any placement, so the defensible comparison uses only rank-pinned traffic. E4.
- METIS only bounds imbalance (`ufactor`); hosts have exactly 8 GPUs. **Fixed**: exact-capacity repair
  + swap refinement, never worse than block by construction.

## 6. Bytes are not bottlenecks; the injection test is tautological — tested by experiment
Injecting extra bytes and detecting them with a bytes metric proves nothing. E5 injects a slow NIC
(no byte change): bytes-only analysis is at chance; time-weighted analysis finds it. Extraction must
record per-event durations (PMPI shim, Chakra and Nsight do; NCCL INFO logs do not).

## 7. Claim 4 needs hardware nobody has arranged — open
Measured speedups need a multi-node cluster you can re-place jobs on. Without one, the paper's claims
must be stated as model/simulation results (all current results are). Options: institutional allocation,
a collaborating lab, or public traces (Chakra ETs, DOE/NERSC MPI traces) + a simulator (ASTRA-sim, SimGrid).

## 8. Scope too wide — open (paper text)
Databases, Flink, microservices, HPC and ML in one paper is shallow everywhere. Paper 1 should cover GPU
training (and optionally MPI stencils as the classic case), with the rest as future work.

## 9. Smaller fixes — fixed in code
- python-louvain is undirected-only → `networkx.community.louvain_communities` on the explicit symmetric view.
- `python-metis` bindings are stale → `pymetis`.
- The two documents disagree (prototype module names, 32 vs 64 workers, 15–40% vs 20–40%) → the repo is
  now the single source of truth.

## 10. Roadmap-level corrections (Papers 2–5)
- C-GAPE: "the access-pattern graph is implicit in the communication graph" is false — a comm trace has no
  object identity. C-GAPE needs its own extraction (`gape/cache.py::AccessTrace`).
- I-GAPE: spectral / edit-distance similarity is permutation-invariant and misses the drift that matters
  for placement (which ranks talk). Use rank-aligned metrics as triggers (I1).
- GAPE: "real time" joint decisions must pay migration cost; the engine is only worth it in a regime that
  G1 quantifies.
