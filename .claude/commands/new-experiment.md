---
description: Scaffold a new experiment script that follows the repo's conventions
argument-hint: "<paper 1-5> <short name> <one-line question>"
---
Create `papers/<paper dir>/experiments/<next id>_<name>.py` for: $ARGUMENTS

Follow the pattern of the existing scripts in that folder:
- module docstring: the question, the claim/hypothesis it tests, setup, and the EXPECTED result
  written before running (so we can tell if we were wrong);
- use `cage.expkit`: `outdir`, `save_json`, `save_fig`, `write_summary` (scenario=True unless the input
  is a real measured trace), and honour `X.quick()` with a smaller sweep;
- baselines: block placement first; report pinned and fixed regimes when placement is involved;
- seeds fixed; no number in the summary that is not computed in the script.
Then add a test in `tests/` if the experiment relies on a new invariant, run it in quick mode, and add a
row to the claims table in the paper's README. Show me the summary.md.
