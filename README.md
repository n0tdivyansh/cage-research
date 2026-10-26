# CAGE → GAPE

Graph analysis of communication in distributed systems: a five-paper research program.
CAGE (Paper 1) extracts communication graphs from traces, tests their structure against null models and
recommends capacity-exact placements; T-GAPE, C-GAPE, I-GAPE and GAPE (Papers 2–5) extend it to network
topology, data caching, iteration-aware re-planning and an adaptive engine.

```bash
pip install -e ".[dev]"
python -m pytest
cage run --quick          # every experiment, small sweeps
```

Start with `CLAUDE.md` (context, rules, findings, open tasks), then `papers/0X-*/README.md`.
All current results come from synthetic workloads and analytic cost models — scenario values, not
hardware measurements.
