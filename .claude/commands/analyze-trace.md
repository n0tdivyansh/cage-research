---
description: Analyze a real trace (NCCL log, Chakra JSON, PMPI output, or CSV) with CAGE
argument-hint: "<path> [gpus-per-node]"
---
1. Identify the trace format of `$ARGUMENTS` (see cage/extract/__init__.py). If it is NCCL or Chakra
   and it is the first real file of that kind, open a sample and check the parser's regexes / field
   names against it before trusting results; fix the parser and add a trimmed copy as a fixture test.
2. Run `cage analyze <path> --gpus-per-node <n> --n-null 20 --out results/real/<name>`.
3. Summarise report.md, flagging: assumptions the trace carries (collective algorithm), whether the
   community verdict is significant, and both placement regimes.
4. This is the first REAL measurement in the project — label it as such and keep it separate from
   scenario results.
