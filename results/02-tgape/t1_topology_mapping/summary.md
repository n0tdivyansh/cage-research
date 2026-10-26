# T1 — Topology-aware mapping

_Generated 2026-10-26 11:00_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

| workload | oversub | strategy | cross_leaf_frac | est_time_s | time_vs_block | max_uplink_ms | max_nic_ms | uplink_bound |
|---|---|---|---|---|---|---|---|---|
| MoE aff 0.6, 128 ranks (pinned) | 1 | block | 0.782 | 3.0482 | 1.000 | 1208.323 | 3025.581 | False |
| MoE aff 0.6, 128 ranks (pinned) | 1 | flat | 0.546 | 1.5186 | 0.498 | 863.862 | 1402.471 | False |
| MoE aff 0.6, 128 ranks (pinned) | 1 | hierarchical | 0.313 | 1.5186 | 0.498 | 454.024 | 1402.471 | False |
| MoE aff 0.6, 128 ranks (pinned) | 1 | top_down | 0.313 | 1.5376 | 0.504 | 454.024 | 1418.966 | False |
| MoE aff 0.6, 128 ranks (pinned) | 2 | block | 0.782 | 3.0482 | 1.000 | 2416.645 | 3025.581 | False |
| MoE aff 0.6, 128 ranks (pinned) | 2 | flat | 0.546 | 1.8438 | 0.605 | 1727.724 | 1402.471 | True |
| MoE aff 0.6, 128 ranks (pinned) | 2 | hierarchical | 0.313 | 1.5186 | 0.498 | 908.048 | 1402.471 | False |
| MoE aff 0.6, 128 ranks (pinned) | 2 | top_down | 0.313 | 1.5376 | 0.504 | 908.048 | 1418.966 | False |
| MoE aff 0.6, 128 ranks (pinned) | 4 | block | 0.782 | 4.8559 | 1.000 | 4833.290 | 3025.581 | True |
| MoE aff 0.6, 128 ranks (pinned) | 4 | flat | 0.546 | 3.5716 | 0.736 | 3455.447 | 1402.471 | True |
| MoE aff 0.6, 128 ranks (pinned) | 4 | hierarchical | 0.313 | 1.9322 | 0.398 | 1816.096 | 1402.471 | True |
| MoE aff 0.6, 128 ranks (pinned) | 4 | top_down | 0.313 | 1.9347 | 0.398 | 1816.096 | 1418.966 | True |
| Stencil 3D 8x4x4 (fixed) | 1 | block | 0.158 | 0.0019 | 1.000 | 0.640 | 1.600 | False |
| Stencil 3D 8x4x4 (fixed) | 1 | flat | 0.158 | 0.0016 | 0.857 | 0.640 | 1.280 | False |
| Stencil 3D 8x4x4 (fixed) | 1 | hierarchical | 0.158 | 0.0016 | 0.857 | 0.640 | 1.280 | False |
| Stencil 3D 8x4x4 (fixed) | 1 | top_down | 0.158 | 0.0016 | 0.857 | 0.480 | 1.280 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | block | 0.158 | 0.0019 | 1.000 | 1.280 | 1.600 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | flat | 0.158 | 0.0016 | 0.857 | 1.280 | 1.280 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | hierarchical | 0.158 | 0.0016 | 0.857 | 1.280 | 1.280 | False |
| Stencil 3D 8x4x4 (fixed) | 2 | top_down | 0.158 | 0.0016 | 0.857 | 0.960 | 1.280 | False |
| Stencil 3D 8x4x4 (fixed) | 4 | block | 0.158 | 0.0028 | 1.000 | 2.560 | 1.600 | True |
| Stencil 3D 8x4x4 (fixed) | 4 | flat | 0.158 | 0.0029 | 1.019 | 2.560 | 1.280 | True |
| Stencil 3D 8x4x4 (fixed) | 4 | hierarchical | 0.158 | 0.0029 | 1.019 | 2.560 | 1.280 | True |
| Stencil 3D 8x4x4 (fixed) | 4 | top_down | 0.158 | 0.0022 | 0.792 | 1.920 | 1.280 | True |
| Megatron 8·4·4, scrambled (fixed) | 1 | block | 0.739 | 5.3155 | 1.000 | 4233.227 | 5214.359 | False |
| Megatron 8·4·4, scrambled (fixed) | 1 | flat | 0.058 | 1.2097 | 0.228 | 394.600 | 408.022 | False |
| Megatron 8·4·4, scrambled (fixed) | 1 | hierarchical | 0.003 | 1.2097 | 0.228 | 21.475 | 408.022 | False |
| Megatron 8·4·4, scrambled (fixed) | 1 | top_down | 0.003 | 1.2097 | 0.228 | 21.475 | 408.022 | False |
| Megatron 8·4·4, scrambled (fixed) | 2 | block | 0.739 | 8.5676 | 1.000 | 8466.454 | 5214.359 | True |
| Megatron 8·4·4, scrambled (fixed) | 2 | flat | 0.058 | 1.5909 | 0.186 | 789.200 | 408.022 | True |
| Megatron 8·4·4, scrambled (fixed) | 2 | hierarchical | 0.003 | 1.2097 | 0.141 | 42.950 | 408.022 | False |
| Megatron 8·4·4, scrambled (fixed) | 2 | top_down | 0.003 | 1.2097 | 0.141 | 42.950 | 408.022 | False |
| Megatron 8·4·4, scrambled (fixed) | 4 | block | 0.739 | 17.0340 | 1.000 | 16932.909 | 5214.359 | True |
| Megatron 8·4·4, scrambled (fixed) | 4 | flat | 0.058 | 2.3801 | 0.140 | 1578.400 | 408.022 | True |
| Megatron 8·4·4, scrambled (fixed) | 4 | hierarchical | 0.003 | 1.2097 | 0.071 | 85.899 | 408.022 | False |
| Megatron 8·4·4, scrambled (fixed) | 4 | top_down | 0.003 | 1.2097 | 0.071 | 85.899 | 408.022 | False |
| MoE aff 0.6, 128 ranks (nccl) | 1 | block | 0.538 | 3.2202 | 1.000 | 1228.166 | 3104.956 | False |
| MoE aff 0.6, 128 ranks (nccl) | 1 | flat | 0.379 | 1.6906 | 0.525 | 794.667 | 1481.846 | False |
| MoE aff 0.6, 128 ranks (nccl) | 1 | hierarchical | 0.221 | 1.6906 | 0.525 | 473.868 | 1481.846 | False |
| MoE aff 0.6, 128 ranks (nccl) | 1 | top_down | 0.221 | 1.7096 | 0.531 | 473.868 | 1498.341 | False |
| MoE aff 0.6, 128 ranks (nccl) | 2 | block | 0.538 | 3.2202 | 1.000 | 2456.333 | 3104.956 | False |
| MoE aff 0.6, 128 ranks (nccl) | 2 | flat | 0.379 | 1.7981 | 0.558 | 1589.333 | 1481.846 | True |
| MoE aff 0.6, 128 ranks (nccl) | 2 | hierarchical | 0.221 | 1.6906 | 0.525 | 947.735 | 1481.846 | False |
| MoE aff 0.6, 128 ranks (nccl) | 2 | top_down | 0.221 | 1.7096 | 0.531 | 947.735 | 1498.341 | False |
| MoE aff 0.6, 128 ranks (nccl) | 4 | block | 0.538 | 5.0279 | 1.000 | 4912.665 | 3104.956 | True |
| MoE aff 0.6, 128 ranks (nccl) | 4 | flat | 0.379 | 3.3874 | 0.674 | 3178.666 | 1481.846 | True |
| MoE aff 0.6, 128 ranks (nccl) | 4 | hierarchical | 0.221 | 2.1042 | 0.419 | 1895.471 | 1481.846 | True |
| MoE aff 0.6, 128 ranks (nccl) | 4 | top_down | 0.221 | 2.1067 | 0.419 | 1895.471 | 1498.341 | True |
| Megatron 8·4·4, scrambled (nccl) | 1 | block | 0.451 | 5.5150 | 1.000 | 2693.750 | 5214.359 | False |
| Megatron 8·4·4, scrambled (nccl) | 1 | flat | 0.048 | 1.2097 | 0.219 | 303.332 | 408.022 | False |
| Megatron 8·4·4, scrambled (nccl) | 1 | hierarchical | 0.003 | 1.2097 | 0.219 | 21.475 | 408.022 | False |
| Megatron 8·4·4, scrambled (nccl) | 1 | top_down | 0.003 | 1.2097 | 0.219 | 21.475 | 408.022 | False |
| Megatron 8·4·4, scrambled (nccl) | 2 | block | 0.451 | 5.6881 | 1.000 | 5387.500 | 5214.359 | True |
| Megatron 8·4·4, scrambled (nccl) | 2 | flat | 0.048 | 1.4084 | 0.248 | 606.664 | 408.022 | True |
| Megatron 8·4·4, scrambled (nccl) | 2 | hierarchical | 0.003 | 1.2097 | 0.213 | 42.950 | 408.022 | False |
| Megatron 8·4·4, scrambled (nccl) | 2 | top_down | 0.003 | 1.2097 | 0.213 | 42.950 | 408.022 | False |
| Megatron 8·4·4, scrambled (nccl) | 4 | block | 0.451 | 11.0756 | 1.000 | 10774.999 | 5214.359 | True |
| Megatron 8·4·4, scrambled (nccl) | 4 | flat | 0.048 | 2.0151 | 0.182 | 1213.328 | 408.022 | True |
| Megatron 8·4·4, scrambled (nccl) | 4 | hierarchical | 0.003 | 1.2097 | 0.109 | 85.899 | 408.022 | False |
| Megatron 8·4·4, scrambled (nccl) | 4 | top_down | 0.003 | 1.2097 | 0.109 | 85.899 | 408.022 | False |
