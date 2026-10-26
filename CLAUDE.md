# CAGE / GAPE research repo — context for Claude Code

Five-paper research program on graph analysis of distributed-system communication.
Owner: Divyansh Verma (B.Tech CSE, NIET). Paper 1 co-authors on the existing draft: Ashutosh Singh,
Aniruddha Badhani. Course context: Research Methodology, Ethics and Writing.

| Paper | System | Question | Code | Experiments |
|---|---|---|---|---|
| 1 | CAGE | Do communication graphs have exploitable structure, and when does graph-aware placement beat the default? | `cage/` | `papers/01-cage/experiments/e1–e7` |
| 2 | T-GAPE | Does mapping above the host level (leaf/spine) matter? | `gape/topology.py` | `papers/02-tgape/experiments/t1` |
| 3 | C-GAPE | Does the data-access graph beat per-host frequency caching? | `gape/cache.py` | `papers/03-cgape/experiments/c1` |
| 4 | I-GAPE | How stable are graphs across iterations; when to re-plan? | `gape/temporal.py` | `papers/04-igape/experiments/i1` |
| 5 | GAPE | Is an adaptive engine worth it once migration is paid for? | `gape/engine.py` | `papers/05-gape/experiments/g1` |

Each `papers/0X-*/README.md` holds that paper's thesis, claims → experiment map, current findings and open tasks.
Original planning docs: `docs/original/`. The critique that reshaped them: `docs/review_findings.md`.
How each open question was resolved: `docs/design_decisions.md`.

Paper drafts (IEEEtran; every number must trace to `results/**/summary.md`, rule E5):
- `papers/01-cage/draft/cage.tex` (+ `cage.pdf`, `refs.bib`) — CAGE, authors Verma, Badhani, Singh.
- `papers/02-tgape/draft/tgape.tex` (+ `tgape.pdf`) — T-GAPE; cites `../../01-cage/draft/refs.bib`.
- Papers 3–5 have no draft yet. Do not edit numbers in a draft without re-running the experiment
  behind it; `papers/01-cage/draft/README.md` still names the old `cage_paper_v2.tex` file.

## Commands
```bash
pip install -e ".[dev]"            # numpy scipy networkx matplotlib pymetis pytest
python -m pytest                   # 38 pass + 1 skipped (PMPI self-test needs MPI); must pass before any result is trusted
cage run --quick                   # all experiments, small sweeps (~1 min)
cage run --paper 1 --only e4       # one experiment, full size
cage synth moe -p world=128 -p affinity=0.6 --out traces/moe.csv
cage analyze traces/moe.csv --gpus-per-node 8 --out results/real/moe
cage analyze nccl_logs/ --format nccl --gpus-per-node 8 --out results/real/job1
```
Results land in `results/<paper>/<experiment>/` as `results.json`, `summary.md`, PNG + PDF figures.
Slash commands: `/run-experiments`, `/check-claims`, `/new-experiment`, `/analyze-trace`.

## Epistemic rules (non-negotiable — the v2 revision was about exactly this)
- **E1** Every number from `cage/workloads.py` or the cost models is a *scenario value*. Say so in prose
  and captions. Only `results/real/` (from real traces) holds measurements. No speedup claims on hardware
  until a real run exists.
- **E2** Never report modularity Q without the null model (z, empirical p, ≥19 null samples).
- **E3** Placement baseline is **block** placement. Round-robin is a strawman; show it only beside block.
  For GPU/NCCL jobs report the `nccl` regime (collectives re-ringed per placement) as the headline.
- **E4** Betweenness is experimental (no physical meaning on app-level graphs); not evidence.
- **E5** No number in a draft without a backing `results/**` file. Run `/check-claims` before sharing a draft.
- **E6** Write the expected result in an experiment's docstring *before* running it; report misses.
- **E7** Novelty: do not claim "no tool builds communication graphs". See review_findings.md §2.

## Architecture in one screen
- `cage/trace.py` — `CommTrace`: event table (src, dst, bytes, msgs, step, op, duration). Static graph,
  per-step matrices and windows are views. `meta['labels']` ground truth; `meta['collectives']` records
  library collectives so placements can re-expand them.
- `cage/collectives.py` — collective → p2p expansion (ring/tree/chain/alltoall/hierarchical).
- `cage/workloads.py` — DP ring, hierarchical DP, Megatron 3D, MoE (skew + affinity), MoE with regime
  changes, stencils, planted partition.
- `cage/metrics.py` — strength, Gini, top-share, robust-z hotspots, Louvain + degree-preserving null,
  experimental betweenness, NMI/ARI.
- `cage/partition.py` — exact-capacity METIS + repair + swap refinement; `method="best"` never worse than block.
- `cage/placement.py` — host cost model; `nccl_matrix` / `recommend_nccl` (NCCL re-ringing model);
  `time_weighted`.
- `cage/extract/` — parsers: csv, pmpi (shim output), chakra (JSON), nccl (debug logs).
- `tools/pmpi_shim/` — C PMPI shim + self-test (**untested**: needs mpicc; `make test`).
- `gape/` — topology (leaf-spine), cache (access graphs), temporal (drift + re-planning), engine.

## Current findings (all scenario values, full run 2026-10-26; source: `results/**/summary.md`)
- E1 (20 null samples per point): planted structure recovered — Louvain NMI 0.99 → 0.51 → 0.27, METIS ARI
  0.99 → 0.33 → 0.12 as mixing μ goes 0.05 → 0.6 → 0.75. Significance z 25 at μ=0.05, 3.3 at 0.6, 0.6 at 0.75:
  z crosses 3 between μ=0.6 and 0.75.
- E2 (20 null samples): ring Q=0.74 (z −1.1), hierarchical-leader Q=0.65 (z −0.8), sparse random Q=0.48 (z 2.1):
  high Q, no structure. Stencil (z 6.2), Megatron (z 10.2), SBM (z 17.4) significant. The "Q > 0.3" rule misfires.
- E3 (20 null samples): H2 (top 20% carry 70–80%) fails everywhere; max top-20% share 0.42 (MoE, Zipf 1.5).
  DP is balanced. Significant structure only in DP hier. rail (z 5.7), stencils (z 6.2–6.8), Megatron (z 10.2);
  skewed MoE (Zipf 1.5) sits *below* its null (z −5.0).
- E4 (NCCL regime, vs block): DP ring, hierarchical DP, Megatron in launcher order → 0% gain (block is already
  right). Gains only where rank order mismatches structure: scrambled Megatron −93% cross-host bytes,
  2D stencil −43%, 3D stencil −25%, MoE with routing affinity −56% (aff 0.6). MoE affinity 0: −0.1% bytes but
  the time model reports −15% with 46 ranks moved — not a byte gain; understand it before citing.
  Affinity sweep: interleaved groups give −19/−38/−56/−75% at aff 0.2/0.4/0.6/0.8; contiguous (block-layout)
  groups give 0% at every affinity — the gain is a rank-order mismatch, not affinity per se.
  Round-robin ties CAGE on interleaved-group MoE and is catastrophic elsewhere (up to 17.6× block's bytes).
- E5: slow NIC (no byte change): bytes-only at chance (0.12), time-weighted 0.75–1.00. Volume ×3 injection is
  not even reliably detected on skewed MoE (0.06) — hot experts dominate.
- E6: stationary routing: placement quality and community NMI 1.0 from 1% of steps (Kendall τ 0.95,
  top-10 Jaccard 0.82 at 1%).
- E7 (machine-dependent): swap refinement is O(n²)/iter — partition_best 10.7 s at 1024 ranks vs 0.79 s at 512;
  Louvain 0.06 s, METIS 0.21 s, betweenness 0.25 s at 1024.
- T1 (time vs block; uplink/NIC = busiest link time): leaf-aware mapping matters only under oversubscription.
  At 1:1 flat = hierarchical (MoE 0.50×, Megatron-scrambled 0.23×) because the NIC, not the uplink, is the
  bottleneck — hierarchical still cuts MoE's busiest uplink 864 → 454 ms. At 4:1: MoE hierarchical 0.40× vs
  flat 0.74× (pinned), 0.42× vs 0.67× (NCCL regime); scrambled Megatron 0.07× vs 0.14× (pinned), 0.11× vs
  0.18× (NCCL). 3D stencil at 4:1: flat/hierarchical 1.02× (worse than block), top_down 0.79× — minimising
  total cut does not minimise the busiest uplink.
- C1: graph-aware data homes alone ≈ LFU (≤1.5 pt under block placement). Joint rank + data placement wins at
  high locality (remote fetch 0.22 block+LFU → 0.09 co-access+LFU → 0.02 joint at affinity 0.9, 5% cache).
  Replication (+rep) adds nothing anywhere. Graph-aware caching only pays *jointly* with placement.
- I1: rank-aligned drift fires at regime boundaries (rel_l1 20×, cosine 584×, partition drift zero off-boundary),
  spectral only 3.2× (permutation-blind). Re-planning: triggered 112 s, periodic 116 s, once 166 s,
  block 202 s, oracle 102 s (upper bound, uses the future).
- G1: adaptive engine 0.53× block at ≤0.01 s/rank migration and 0.63× at 0.1 s/rank (once: 0.89–0.91×);
  topology-oblivious GAPE is within 1–2% of GAPE. At ≥1 s/rank the cost gate declines every move (=block)
  while ungated re-planning costs 1.27–4.10× block.

## Open tasks (priority order)
1. **Real traces.** Collect NCCL logs (`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL`) or Chakra ETs from any
   multi-GPU job (even 2×2 GPUs on Kaggle/Colab-class hardware is useful for parser validation). Verify the
   nccl/chakra parsers against them; add trimmed real files as fixtures. Public Chakra/MLCommons traces are
   an alternative.
2. **Compile and run the PMPI shim** (`cd tools/pmpi_shim && make test`) on any machine with OpenMPI/MPICH;
   then run NAS Parallel Benchmarks (CG, MG, LU) under it for real HPC traces.
3. **Drafts**: run `/check-claims` on `papers/01-cage/draft/cage.tex` and `papers/02-tgape/draft/tgape.tex`
   after every re-run. Paper 1 still needs a case study from `cage analyze` output on a real trace (task 1).
4. Speed up `partition.refine_swaps` (incremental gain updates or restrict to boundary vertices) for ≥1024 ranks.
5. Hardware or simulator path for Claim 4: ASTRA-sim / SimGrid replay of placements, or a cluster allocation.
6. C-GAPE: wire cache planning into `gape/engine.py` (currently not in the loop).
7. Related work: `papers/01-cage/draft/refs.bib` exists (shared by T-GAPE); verify every entry against
   review_findings §2 and the original sources.
