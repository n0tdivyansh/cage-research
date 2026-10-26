# E3 — Workload characterisation

_Generated 2026-10-26 10:38_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

| workload | ranks | density | gini | top20_share | max/median | hotspots | Q | z | H2_holds(top20>=0.7) |
|---|---|---|---|---|---|---|---|---|---|
| DP ring | 64 | 0.03 | 0.00 | 0.20 | 1.00 | 0 | 0.74 | -1.20 | False |
| DP hier. leader | 64 | 0.03 | 0.10 | 0.25 | 1.38 | 0 | 0.75 | 0.02 | False |
| DP hier. rail | 64 | 0.06 | 0.00 | 0.20 | 1.00 | 0 | 0.76 | 7.08 | False |
| Megatron 8·4·2 | 64 | 0.07 | 0.00 | 0.20 | 1.00 | 0 | 0.82 | 9.89 | False |
| MoE zipf 0 | 64 | 1.00 | 0.01 | 0.21 | 1.02 | 0 | 0.23 | 2.05 | False |
| MoE zipf 1.0 | 64 | 1.00 | 0.19 | 0.35 | 6.00 | 6 | 0.22 | -0.29 | False |
| MoE zipf 1.5 | 64 | 1.00 | 0.27 | 0.42 | 12.60 | 11 | 0.22 | -4.47 | False |
| MoE zipf 1 aff 0.6 | 64 | 1.00 | 0.15 | 0.32 | 3.97 | 6 | 0.28 | 2.42 | False |
| Stencil 2D 8x8 | 64 | 0.06 | 0.09 | 0.23 | 1.00 | 0 | 0.59 | 7.31 | False |
| Stencil 3D 4x4x4 | 64 | 0.07 | 0.10 | 0.25 | 1.33 | 0 | 0.54 | 6.03 | False |
