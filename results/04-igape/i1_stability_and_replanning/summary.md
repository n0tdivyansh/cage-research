# I1 — Stability and re-planning

_Generated 2026-10-26 10:57_

> **Scenario values.** Inputs are synthetic workload models and costs come from an analytic model. These are not measurements on hardware.

## Drift signal at regime boundaries (mean drift at boundary / mean elsewhere)

- rel_l1: 20.0x
- cosine: 584.4x
- spectral: 3.2x
- partition: infx

## Re-planning policies

| migration_s_per_rank | policy | param | comm_s | migration_s | total_s | replans | ranks_moved |
|---|---|---|---|---|---|---|---|
| 0.000 | static_block | None | 201.767 | 0.000 | 201.767 | 0 | 0 |
| 0.000 | once | None | 166.494 | 0.000 | 166.494 | 1 | 56 |
| 0.000 | periodic | 10 | 116.155 | 0.000 | 116.155 | 12 | 192 |
| 0.000 | periodic | 30 | 116.155 | 0.000 | 116.155 | 4 | 192 |
| 0.000 | triggered | 0.1 | 112.237 | 0.000 | 112.237 | 16 | 192 |
| 0.000 | triggered | 0.3 | 112.237 | 0.000 | 112.237 | 4 | 192 |
| 0.000 | oracle | None | 101.727 | 0.000 | 101.727 | 120 | 192 |
| 0.002 | static_block | None | 201.767 | 0.000 | 201.767 | 0 | 0 |
| 0.002 | once | None | 166.494 | 0.112 | 166.606 | 1 | 56 |
| 0.002 | periodic | 10 | 116.155 | 0.384 | 116.539 | 12 | 192 |
| 0.002 | periodic | 30 | 116.155 | 0.384 | 116.539 | 4 | 192 |
| 0.002 | triggered | 0.1 | 112.237 | 0.384 | 112.621 | 16 | 192 |
| 0.002 | triggered | 0.3 | 112.237 | 0.384 | 112.621 | 4 | 192 |
| 0.002 | oracle | None | 101.727 | 0.384 | 102.111 | 120 | 192 |
| 0.010 | static_block | None | 201.767 | 0.000 | 201.767 | 0 | 0 |
| 0.010 | once | None | 166.494 | 0.560 | 167.054 | 1 | 56 |
| 0.010 | periodic | 10 | 116.155 | 1.920 | 118.075 | 12 | 192 |
| 0.010 | periodic | 30 | 116.155 | 1.920 | 118.075 | 4 | 192 |
| 0.010 | triggered | 0.1 | 112.237 | 1.920 | 114.157 | 16 | 192 |
| 0.010 | triggered | 0.3 | 112.237 | 1.920 | 114.157 | 4 | 192 |
| 0.010 | oracle | None | 101.727 | 1.920 | 103.647 | 120 | 192 |

oracle re-plans on the step's own matrix (future knowledge) — an upper bound, not a policy.
