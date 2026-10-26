# E4 — Placement benefit vs. block placement

_Generated 2026-10-26 10:56_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

## Regime B (NCCL model: collectives re-ringed per placement — defensible for GPU jobs)

| workload | placement | cross_host_frac | cross_reduction_vs_block | time_reduction_vs_block | ranks_moved |
|---|---|---|---|---|---|
| DP ring | round_robin | 0.125 | 0.000 | 0.000 | 56 |
| DP ring | cage | 0.125 | 0.000 | 0.000 | 0 |
| DP hier. leader | round_robin | 0.889 | -7.000 | -2.918 | 56 |
| DP hier. leader | cage | 0.111 | 0.000 | 0.000 | 0 |
| Megatron 8·4·2 | round_robin | 0.946 | -16.569 | -3.493 | 56 |
| Megatron 8·4·2 | cage | 0.054 | 0.000 | 0.000 | 0 |
| Megatron, scrambled ranks | round_robin | 0.658 | 0.111 | 0.197 | 56 |
| Megatron, scrambled ranks | cage | 0.054 | 0.927 | 0.787 | 48 |
| Stencil 2D 8x8 | round_robin | 0.500 | 0.000 | 0.000 | 56 |
| Stencil 2D 8x8 | cage | 0.286 | 0.429 | 0.273 | 40 |
| Stencil 3D 4x4x4 | round_robin | 0.667 | -0.500 | -0.286 | 56 |
| Stencil 3D 4x4x4 | cage | 0.333 | 0.250 | 0.314 | 32 |
| MoE aff 0 | round_robin | 0.646 | -0.000 | 0.117 | 56 |
| MoE aff 0 | cage | 0.645 | 0.001 | 0.149 | 46 |
| MoE aff 0.6 | round_robin | 0.296 | 0.563 | 0.488 | 56 |
| MoE aff 0.6 | cage | 0.296 | 0.563 | 0.488 | 56 |

## Regime A (all traffic, collectives fixed in rank order)

| workload | placement | cross_host_frac | cross_reduction_vs_block | time_reduction_vs_block | ranks_moved |
|---|---|---|---|---|---|
| DP ring | round_robin | 1.000 | -7.000 | -2.615 | 56 |
| DP ring | cage | 0.125 | 0.000 | 0.000 | 24 |
| DP hier. leader | round_robin | 0.889 | -7.000 | -2.916 | 56 |
| DP hier. leader | cage | 0.111 | 0.000 | 0.000 | 0 |
| Megatron 8·4·2 | round_robin | 0.946 | -16.569 | -3.082 | 56 |
| Megatron 8·4·2 | cage | 0.054 | 0.000 | 0.000 | 0 |
| Megatron, scrambled ranks | round_robin | 0.865 | 0.026 | -0.000 | 56 |
| Megatron, scrambled ranks | cage | 0.054 | 0.939 | 0.766 | 48 |
| Stencil 2D 8x8 | round_robin | 0.500 | 0.000 | 0.000 | 56 |
| Stencil 2D 8x8 | cage | 0.286 | 0.429 | 0.247 | 40 |
| Stencil 3D 4x4x4 | round_robin | 0.667 | -0.500 | -0.258 | 56 |
| Stencil 3D 4x4x4 | cage | 0.333 | 0.250 | 0.294 | 32 |
| MoE aff 0 | round_robin | 0.924 | -0.430 | -0.091 | 56 |
| MoE aff 0 | cage | 0.646 | 0.000 | 0.000 | 0 |
| MoE aff 0.6 | round_robin | 0.587 | 0.133 | 0.249 | 56 |
| MoE aff 0.6 | cage | 0.587 | 0.133 | 0.249 | 56 |

## MoE affinity sweep (NCCL model, CAGE vs block)

| affinity | layout | cross_reduction_vs_block | time_reduction_vs_block | ranks_moved |
|---|---|---|---|---|
| 0.000 | interleaved | 0.001 | 0.149 | 46 |
| 0.000 | block | 0.001 | 0.149 | 46 |
| 0.200 | interleaved | 0.188 | 0.236 | 56 |
| 0.200 | block | 0.000 | 0.000 | 0 |
| 0.400 | interleaved | 0.375 | 0.352 | 56 |
| 0.400 | block | 0.000 | 0.000 | 0 |
| 0.600 | interleaved | 0.563 | 0.488 | 56 |
| 0.600 | block | 0.000 | 0.000 | 0 |
| 0.800 | interleaved | 0.751 | 0.629 | 56 |
| 0.800 | block | 0.000 | 0.000 | 0 |
