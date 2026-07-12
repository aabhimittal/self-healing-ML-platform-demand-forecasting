"""End-to-end self-healing scenario used by both the CLI and the example script.

The scenario walks a single demand series through four monitoring cycles that
exercise every remediation path:

    cycle 1  healthy traffic            -> monitor (no action)
    cycle 2  seasonality shift          -> diagnose + retrain + promote (heal)
    cycle 3  promotion spike            -> diagnose + widen intervals (no retrain)
    cycle 4  robotics slowdown          -> diagnose + renegotiate latency SLO

Then it demonstrates a one-call rollback restoring the previous production model.
Everything runs with the rule-based explainer by default, so it needs no API key.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .config import DEFAULT_CONFIG
from .data.generator import DriftEvent, generate_demand, DemandSeries
from .healing.orchestrator import SelfHealingOrchestrator, CycleReport


@dataclass
class ScenarioResult:
    orchestrator: SelfHealingOrchestrator
    series: DemandSeries
    cycles: List[CycleReport] = field(default_factory=list)
    rolled_back_to: Optional[int] = None


def _latencies(rng: random.Random, mean: float, spread: float, n: int = 200) -> List[float]:
    return [max(1.0, rng.gauss(mean, spread)) for _ in range(n)]


def run_scenario(seed: int = 7, use_llm: Optional[bool] = None, verbose: bool = True) -> ScenarioResult:
    rng = random.Random(seed)

    # One series carrying all three drift scenarios in disjoint day ranges.
    events = [
        DriftEvent("seasonality_shift", start=120, end=150, magnitude=1.0),
        DriftEvent("promotion", start=150, end=165, magnitude=1.0),
        DriftEvent("robotics_slowdown", start=180, end=210, magnitude=1.0),
    ]
    # Near-stationary baseline (small trend, modest yearly amplitude) so that a
    # *healthy* window does not trip the statistical detectors — drift then comes
    # from the injected events, not from the baseline trend.
    series = generate_demand(
        n_days=210,
        base=100.0,
        trend=0.02,
        weekly_amplitude=18.0,
        yearly_amplitude=8.0,
        noise=4.0,
        events=events,
        seed=seed,
    )

    orch = SelfHealingOrchestrator(config=DEFAULT_CONFIG, use_llm=use_llm)

    def log(msg: str = "") -> None:
        if verbose:
            print(msg)

    log("=" * 74)
    log("  SELF-HEALING ML PLATFORM — DEMAND FORECASTING")
    log("=" * 74)

    boot = orch.bootstrap(series, train_end=90)
    log(f"\n[bootstrap] trained + promoted model v{boot.version} "
        f"(baseline MAE={boot.metrics['mae']:.3f}) to production\n")

    cycles_spec = [
        ("healthy traffic", 90, 120, _latencies(rng, 78, 14)),
        ("seasonality shift", 120, 150, _latencies(rng, 82, 16)),
        ("promotion spike", 150, 180, _latencies(rng, 85, 15)),
        ("robotics slowdown", 180, 210, _latencies(rng, 235, 45)),
    ]

    results: List[CycleReport] = []
    for label, start, end, lats in cycles_spec:
        rep = orch.run_cycle(series, start, end, lats)
        results.append(rep)
        _print_cycle(log, label, rep)

    # --- rollback demonstration -------------------------------------------
    prod_before = orch.registry.production().version
    restored = orch.rollback(reason="demo_rollback")
    rolled_to = restored.version if restored else None
    log("-" * 74)
    if restored:
        log(f"[rollback] production reverted v{prod_before} -> v{restored.version} "
            f"in one call (rollback-in-minutes)")
    else:
        log("[rollback] nothing to roll back to")
    log("-" * 74)

    _print_summary(log, orch, results)

    return ScenarioResult(orchestrator=orch, series=series, cycles=results, rolled_back_to=rolled_to)


def _print_cycle(log, label: str, rep: CycleReport) -> None:
    log("-" * 74)
    log(f"[cycle {rep.cycle}] {label}")
    log(f"    drift      : severity={rep.drift.severity} psi={rep.drift.psi} "
        f"ks={rep.drift.ks} mean_shift={rep.drift.mean_shift_pct}%")
    log(f"    confidence : score={rep.confidence.score} status={rep.confidence.status} "
        f"coverage={rep.confidence.coverage}")
    log(f"    latency SLO: {rep.slo.action} (p95 target {rep.slo.new_p95_ms:.0f}ms, "
        f"observed p95 {rep.slo.observed_p95_ms:.0f}ms)")
    if rep.diagnosis:
        d = rep.diagnosis
        log(f"    diagnosis  : cause={d.cause} conf={d.confidence:.2f} src={d.source}")
        log(f"                 \"{d.rationale}\"")
        log(f"    action     : {rep.action}")
        if rep.notes:
            log(f"                 -> {rep.notes}")
    else:
        log("    diagnosis  : (none — no actionable drift)")
    log(f"    production : v{rep.production_version}  healed={rep.healed}")


def _print_summary(log, orch: SelfHealingOrchestrator, results) -> None:
    snap = orch.metrics.snapshot()
    heals = sum(1 for r in results if r.healed)
    log("\n" + "=" * 74)
    log("  EPISODE SUMMARY")
    log("=" * 74)
    log(f"  cycles run          : {len(results)}")
    log(f"  models registered   : {len(orch.registry.versions())}")
    log(f"  retrains            : {int(_get(snap, 'shml_retrains_total'))}")
    log(f"  promotions          : {int(_get(snap, 'shml_promotions_total'))}")
    log(f"  rollbacks           : {int(_get(snap, 'shml_rollbacks_total'))}")
    log(f"  SLO renegotiations  : {int(_get(snap, 'shml_slo_renegotiations_total'))}")
    log(f"  audit events        : {len(orch.audit)}")
    log(f"  human interventions : 0 (fully automated)")
    log("=" * 74)


def _get(snap, name: str) -> float:
    series = snap.get(name, {})
    return sum(series.values()) if isinstance(series, dict) else 0.0
