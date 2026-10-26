# T1 — Topology-aware mapping

_Generated 2026-10-26 10:39_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

| workload | oversub | strategy | cross_leaf_frac | est_time_s | time_vs_block | uplink_bound |
|---|---|---|---|---|---|---|
| MoE aff 0.6, 128 ranks (pinned) | 1 | block | 0.782 | 3.0482 | 1.000 | False |
| MoE aff 0.6, 128 ranks (pinned) | 1 | flat | 0.546 | 1.5186 | 0.498 | False |
| MoE aff 0.6, 128 ranks (pinned) | 1 | hierarchical | 0.313 | 1.5186 | 0.498 | False |
| MoE aff 0.6, 128 ranks (pinned) | 1 | top_down | 0.313 | 1.5376 | 0.504 | False |
| MoE aff 0.6, 128 ranks (pinned) | 2 | block | 0.782 | 3.0482 | 1.000 | False |
| MoE aff 0.6, 128 ranks (pinned) | 2 | flat | 0.546 | 1.8438 | 0.605 | True |
| MoE aff 0.6, 128 ranks (pinned) | 2 | hierarchical | 0.313 | 1.5186 | 0.498 | False |
| MoE aff 0.6, 128 ranks (pinned) | 2 | top_down | 0.313 | 1.5376 | 0.504 | False |
| MoE aff 0.6, 128 ranks (pinned) | 4 | block | 0.782 | 4.8559 | 1.000 | True |
| MoE aff 0.6, 128 ranks (pinned) | 4 | flat | 0.546 | 3.5716 | 0.736 | True |
| MoE aff 0.6, 128 ranks (pinned) | 4 | hierarchical | 0.313 | 1.9322 | 0.398 | True |
| MoE aff 0.6, 128 ranks (pinned) | 4 | top_down | 0.313 | 1.9347 | 0.398 | True |
| Stencil 3D 8x4x4 (fixed) | 1 | block | 0.158 | 0.0019 | 1.000 | False |
| Stencil 3D 8x4x4 (fixed) | 1 | flat | 0.158 | 0.0016 | 0.857 | False |
| Stencil 3D 8x4x4 (fixed) | 1 | hierarchical | 0.158 | 0.0016 | 0.857 | False |
| Stencil 3D 8x4x4 (fixed) | 1 | top_down | 0.158 | 0.0016 | 0.857 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | block | 0.158 | 0.0019 | 1.000 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | flat | 0.158 | 0.0016 | 0.857 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | hierarchical | 0.158 | 0.0016 | 0.857 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | top_down | 0.158 | 0.0016 | 0.857 | False |
| Stencil 3D 8x4x4 (fixed) | 4 | block | 0.158 | 0.0028 | 1.000 | True |
| Stencil 3D 8x4x4 (fixed) | 4 | flat | 0.158 | 0.0029 | 1.019 | True |
| Stencil 3D 8x4x4 (fixed) | 4 | hierarchical | 0.158 | 0.0029 | 1.019 | True |
| Stencil 3D 8x4x4 (fixed) | 4 | top_down | 0.158 | 0.0022 | 0.792 | True |
| Megatron 8·4·4, scrambled (fixed) | 1 | block | 0.739 | 5.3155 | 1.000 | False |
| Megatron 8·4·4, scrambled (fixed) | 1 | flat | 0.058 | 1.2097 | 0.228 | False |
| Megatron 8·4·4, scrambled (fixed) | 1 | hierarchical | 0.003 | 1.2097 | 0.228 | False |
| Megatron 8·4·4, scrambled (fixed) | 1 | top_down | 0.003 | 1.2097 | 0.228 | False |
| Megatron 8·4·4, scrambled (fixed) | 2 | block | 0.739 | 8.5676 | 1.000 | True |
| Megatron 8·4·4, scrambled (fixed) | 2 | flat | 0.058 | 1.5909 | 0.186 | True |
| Megatron 8·4·4, scrambled (fixed) | 2 | hierarchical | 0.003 | 1.2097 | 0.141 | False |
| Megatron 8·4·4, scrambled (fixed) | 2 | top_down | 0.003 | 1.2097 | 0.141 | False |
| Megatron 8·4·4, scrambled (fixed) | 4 | block | 0.739 | 17.0340 | 1.000 | True |
| Megatron 8·4·4, scrambled (fixed) | 4 | flat | 0.058 | 2.3801 | 0.140 | True |
| Megatron 8·4·4, scrambled (fixed) | 4 | hierarchical | 0.003 | 1.2097 | 0.071 | False |
| Megatron 8·4·4, scrambled (fixed) | 4 | top_down | 0.003 | 1.2097 | 0.071 | False |
