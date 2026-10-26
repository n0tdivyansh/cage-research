# Paper 2 — T-GAPE: topology-aware mapping

**Question.** CAGE maps ranks to hosts. Does mapping hosts onto leaves (and leaves onto the spine) add
benefit for ML traffic, and under what oversubscription?

**Known prior art (do not re-claim):** hierarchical / topology mapping of communication graphs —
Hoefler & Snir (ICS'11), TreeMatch, Scotch, LibTopoMap; TopoOpt (NSDI'23) co-designs topology for ML.
T-GAPE's angle: how much the leaf level matters for modern ML traffic under realistic oversubscription,
and whether two-level greedy mapping is enough.

**Model.** `gape/topology.py`: leaf-spine, 4 hosts/leaf, NIC and aggregated uplink bandwidths, ideal ECMP.
Strategies: block, flat (CAGE host partition, oblivious above host), hierarchical (bottom-up), top_down.

## Findings (T1, scenario values)
- 1:1 spine: leaf mapping adds nothing beyond host partitioning (MoE flat = hierarchical = 0.50× block).
- 4:1 spine: hierarchical 0.40× block vs flat 0.74× (MoE); Megatron scrambled 0.07× vs 0.14×.
- Stencil 3D at 4:1: bottom-up hierarchical is slightly *worse* than block (1.02×); top-down 0.79×.
  The two-level heuristic choice matters — an open sub-question.

## Open tasks
- Real topology: fat-tree with 3 tiers, rail-optimised GPU clusters (per-GPU NICs), dragonfly.
- Link-level congestion model (per-spine ECMP collisions) instead of one aggregated uplink.
- Validate against a packet/flow simulator (ASTRA-sim network backend, SimGrid).
