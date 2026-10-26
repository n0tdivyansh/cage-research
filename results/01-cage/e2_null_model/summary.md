# E2 — Modularity needs a null model

_Generated 2026-10-26 10:38_

| graph | Q | null_Q | z | p | naive_rule_Q>0.3 | significant_z>3 |
|---|---|---|---|---|---|---|
| ring (64) | 0.740 | 0.756 | -1.1 | 1.000 | True | False |
| random sparse, deg 4 (64) | 0.482 | 0.445 | 2.1 | 0.095 | True | False |
| random dense, deg 32 (64) | 0.080 | 0.089 | -1.9 | 0.952 | False | False |
| complete, uniform (64) | 0.000 | 0.000 | 0.0 | 1.000 | False | False |
| 2D stencil 8x8 | 0.594 | 0.500 | 6.2 | 0.048 | True | True |
| hier. leader allreduce (32) | 0.653 | 0.677 | -0.8 | 0.714 | True | False |
| SBM, mu≈0.15 (64) | 0.685 | 0.445 | 17.4 | 0.048 | True | True |
| Megatron TP8·PP4·DP2 | 0.821 | 0.661 | 10.2 | 0.048 | True | True |
| MoE, no affinity (64) | 0.224 | 0.225 | -0.1 | 0.571 | False | False |
