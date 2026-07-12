"""Command-line entry point for the self-healing ML platform.

Usage:
    python -m self_healing_ml.cli demo        # run the end-to-end scenario
    python -m self_healing_ml.cli metrics      # print the Prometheus exposition
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .scenario import run_scenario


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="self_healing_ml", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run the end-to-end self-healing scenario")
    demo.add_argument("--seed", type=int, default=7, help="random seed for the generator")
    demo.add_argument("--llm", action="store_true", help="force the LLM diagnosis path")
    demo.add_argument("--no-llm", action="store_true", help="force the rule-based fallback")

    metrics = sub.add_parser("metrics", help="run the scenario and print Prometheus metrics")
    metrics.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)

    if args.command in (None, "demo"):
        use_llm = True if getattr(args, "llm", False) else (False if getattr(args, "no_llm", False) else None)
        result = run_scenario(seed=getattr(args, "seed", 7), use_llm=use_llm, verbose=True)
        return 0 if result.orchestrator.registry.production() else 1

    if args.command == "metrics":
        result = run_scenario(seed=args.seed, verbose=False)
        sys.stdout.write(result.orchestrator.metrics.render())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
