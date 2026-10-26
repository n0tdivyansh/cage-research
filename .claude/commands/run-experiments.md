---
description: Run experiments for one or all papers and summarise what changed
argument-hint: "[paper 1-5 | all] [--quick] [--only e1,e4]"
---
Run `cage run --paper ${ARGUMENTS:-all}` (add `--quick` if I asked for a fast pass).
Then read every `results/<paper>/<exp>/summary.md` that was regenerated and report:
1. which experiments ran and whether any failed (with the traceback cause),
2. the headline number of each experiment, and whether it moved compared with the
   "Current findings" section of CLAUDE.md,
3. any result that contradicts a claim in `papers/<paper>/README.md`.
Do not edit CLAUDE.md or paper READMEs until I confirm the new numbers.
Remember rule E1: every number here is a scenario value from synthetic inputs and an analytic model.
