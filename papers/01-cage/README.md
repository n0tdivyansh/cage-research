# Paper 1 — CAGE: Communication Analysis Graph Engine

**Revised thesis.** Communication-graph structure in distributed training predicts *when*
communication-aware placement pays off: significant community structure (against a null model) whose
groups disagree with the launcher's rank order. Where rank order already matches structure (DP, Megatron
in default order) the default is optimal and graph analysis should say so.

**Contribution (defensible):** (1) a trace → graph → placement pipeline that handles NCCL and MPI
collectives correctly; (2) an empirical characterisation across DP, 3D-parallel, MoE and HPC stencils,
including negative results; (3) evidence that bytes-only graphs cannot locate time bottlenecks.

## Claims → evidence
| Claim | Experiment | Status (scenario values) |
|---|---|---|
| C1 Structure exists in some workloads, not all; raw Q misleads | E2, E3 | Supported: Megatron/stencils z 6–10; rings/hier-DP not significant |
| C2 Existing tools miss it — **reworded**: position against MPIPP, TreeMatch, Hoefler & Snir, ScalAna | — | Paper text |
| C3 Detected structure corresponds to bottlenecks | E1, E5 | Partial: structure recovery yes; bottlenecks need durations, not bytes |
| C4 Placement reduces cross-node traffic vs **block** | E4 | Only where rank order mismatches structure (−25 to −93%); 0% for DP/Megatron default |
| OQ3 How much trace is needed | E6 | 1% of steps suffices when routing is stationary |
| Overhead | E7 | Louvain/METIS < 0.3 s at 1024 ranks; refinement 23 s (fix) |
| H2 80/20 concentration | E3 | Refuted for all modelled workloads |

## Draft
Put the existing drafts in `draft/` (`cage_paper_v2.tex`, `CAGE.pdf`). They predate the review; expect
`/check-claims` to flag the case study, H1–H3 wording, the novelty claim and betweenness usage.
