# E5 — Detection validity

_Generated 2026-10-26 10:56_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

| workload | anomaly | metric | hit_rate@1 | mean_rank_found |
|---|---|---|---|---|
| MoE aff 0 | volume x3 (tautological) | bytes strength | 0.06 | 4.1 |
| MoE aff 0 | NIC at 25% bandwidth | bytes strength | 0.12 | nan |
| MoE aff 0 | NIC at 25% bandwidth | time-weighted strength | 1.00 | nan |
| MoE aff 0 | NIC at 25% bandwidth | (check) slowed host is true bottleneck | 1.00 | nan |
| Megatron 8·4·2 | volume x3 (tautological) | bytes strength | 1.00 | 1.0 |
| Megatron 8·4·2 | NIC at 25% bandwidth | bytes strength | 0.12 | nan |
| Megatron 8·4·2 | NIC at 25% bandwidth | time-weighted strength | 1.00 | nan |
| Megatron 8·4·2 | NIC at 25% bandwidth | (check) slowed host is true bottleneck | 1.00 | nan |
| Stencil 2D 8x8 | volume x3 (tautological) | bytes strength | 0.97 | 2.1 |
| Stencil 2D 8x8 | NIC at 25% bandwidth | bytes strength | 0.12 | nan |
| Stencil 2D 8x8 | NIC at 25% bandwidth | time-weighted strength | 0.75 | nan |
| Stencil 2D 8x8 | NIC at 25% bandwidth | (check) slowed host is true bottleneck | 1.00 | nan |

Chance level for the NIC test is 1/#hosts = 0.125.
Interpretation: a bytes-only graph cannot attribute slowdowns that do not change volume. Real extraction must record per-event durations (the PMPI shim, Chakra and Nsight do).
