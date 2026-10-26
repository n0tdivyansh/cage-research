"""Command line interface.

  cage synth  WORKLOAD [--param k=v ...] --out trace.csv
  cage analyze TRACE [--format auto|csv|pmpi|chakra|nccl] --gpus-per-node 8 --out results/x
  cage run    [--paper 1|2|3|4|5|all] [--only e1,e4] [--quick]
"""
from __future__ import annotations

import argparse
import ast
import os
import runpy
import sys
from pathlib import Path

from . import workloads as W
from .expkit import ROOT

PAPER_DIRS = {"1": "01-cage", "2": "02-tgape", "3": "03-cgape", "4": "04-igape", "5": "05-gape"}


def _parse_params(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="generate a synthetic workload trace")
    s.add_argument("workload", choices=sorted(W.REGISTRY))
    s.add_argument("--param", "-p", action="append", help="generator kwarg, e.g. -p world=128")
    s.add_argument("--out", required=True)
    a = sub.add_parser("analyze", help="analyze a trace")
    a.add_argument("trace")
    a.add_argument("--format", default="auto", choices=["auto", "csv", "pmpi", "chakra", "nccl"])
    a.add_argument("--gpus-per-node", type=int, default=8)
    a.add_argument("--algo", default="ring", help="collective algorithm assumed when expanding")
    a.add_argument("--n-null", type=int, default=20)
    a.add_argument("--betweenness", action="store_true", help="include experimental betweenness")
    a.add_argument("--out", required=True)
    r = sub.add_parser("run", help="run experiment scripts")
    r.add_argument("--paper", default="all")
    r.add_argument("--only", default=None, help="comma-separated script prefixes, e.g. e1,e4")
    r.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "synth":
        t = W.REGISTRY[args.workload](**_parse_params(args.param))
        t.to_csv(args.out)
        print(f"wrote {args.out}: {t.n_ranks} ranks, {len(t)} events, {t.n_steps} steps")
        return 0
    if args.cmd == "analyze":
        from .extract import load
        from .report import analyze
        kw = {"algo": args.algo} if args.format in ("pmpi", "chakra", "nccl") else {}
        t = load(args.trace, args.format, **kw)
        analyze(t, args.gpus_per_node, args.out, n_null=args.n_null, betweenness=args.betweenness)
        print((Path(args.out) / "report.md").read_text())
        return 0
    if args.cmd == "run":
        if args.quick:
            os.environ["CAGE_QUICK"] = "1"
        papers = PAPER_DIRS.values() if args.paper == "all" else [PAPER_DIRS[p] for p in args.paper.split(",")]
        only = args.only.split(",") if args.only else None
        sys.path.insert(0, str(ROOT))
        failed = []
        for pd in papers:
            for script in sorted((ROOT / "papers" / pd / "experiments").glob("*.py")):
                if only and not any(script.name.startswith(o) for o in only):
                    continue
                print(f"\n=== {pd}/{script.name}", flush=True)
                try:
                    runpy.run_path(str(script), run_name="__main__")
                except Exception as e:  # keep going; report at the end
                    failed.append((script.name, repr(e)))
                    print(f"FAILED: {e!r}")
        if failed:
            print("\nFailures:", *failed, sep="\n  ")
            return 1
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
