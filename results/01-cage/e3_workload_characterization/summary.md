# E3 — Workload characterisation

_Generated 2026-10-26 10:59_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

| workload | ranks | density | gini | top20_share | max/median | hotspots | Q | z | H2_holds(top20>=0.7) |
|---|---|---|---|---|---|---|---|---|---|
| DP ring | 64 | 0.03 | 0.00 | 0.20 | 1.00 | 0 | 0.74 | -1.14 | False |
| DP hier. leader | 64 | 0.03 | 0.10 | 0.25 | 1.38 | 0 | 0.75 | 0.00 | False |
| DP hier. rail | 64 | 0.06 | 0.00 | 0.20 | 1.00 | 0 | 0.76 | 5.73 | False |
| Megatron 8·4·2 | 64 | 0.07 | 0.00 | 0.20 | 1.00 | 0 | 0.82 | 10.23 | False |
| MoE zipf 0 | 64 | 1.00 | 0.01 | 0.21 | 1.02 | 0 | 0.23 | 2.25 | False |
| MoE zipf 1.0 | 64 | 1.00 | 0.19 | 0.35 | 6.00 | 6 | 0.22 | -0.13 | False |
| MoE zipf 1.5 | 64 | 1.00 | 0.27 | 0.42 | 12.60 | 11 | 0.22 | -4.97 | False |
| MoE zipf 1 aff 0.6 | 64 | 1.00 | 0.15 | 0.32 | 3.97 | 6 | 0.28 | 2.48 | False |
| Stencil 2D 8x8 | 64 | 0.06 | 0.09 | 0.23 | 1.00 | 0 | 0.59 | 6.24 | False |
| Stencil 3D 4x4x4 | 64 | 0.07 | 0.10 | 0.25 | 1.33 | 0 | 0.54 | 6.82 | False |
