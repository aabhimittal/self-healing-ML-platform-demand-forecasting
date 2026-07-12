#!/usr/bin/env python3
"""Run the end-to-end self-healing scenario.

    python examples/run_demo.py            # rule-based diagnosis (no API key needed)
    python examples/run_demo.py --llm      # use Claude for semantic drift diagnosis

The LLM path requires the `anthropic` package and an ANTHROPIC_API_KEY; without
them the platform automatically falls back to the deterministic rule-based
explainer, so this script always produces a full timeline.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `src/` importable when run straight from a checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from self_healing_ml.scenario import run_scenario  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--llm", action="store_true", help="force the Claude diagnosis path")
    parser.add_argument("--no-llm", action="store_true", help="force the rule-based fallback")
    args = parser.parse_args()

    use_llm = True if args.llm else (False if args.no_llm else None)
    run_scenario(seed=args.seed, use_llm=use_llm, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
