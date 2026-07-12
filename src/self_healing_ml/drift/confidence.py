"""Model 'confidence decay' score.

A single number in ``[0, 1]`` that captures how much a *deployed* model's
reliability has eroded since it went live, combining two independent signals:

    1. **Error growth** — how much recent forecast error has grown relative to
       the error the model achieved at validation time.
    2. **Interval breach** — how badly the model's prediction intervals are
       failing to cover reality (a well-calibrated 90% interval should contain
       ~90% of actuals).

0.0 means "as trustworthy as at deployment"; 1.0 means "confidence collapsed".
The self-healing orchestrator uses this to decide when to retrain even if the
input distribution alone hasn't tripped a statistical drift alert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..config import ConfidenceThresholds


@dataclass
class ConfidenceReport:
    score: float  # 0 healthy .. 1 collapsed
    status: str   # "healthy" | "watch" | "heal"
    error_growth: float
    coverage: float
    baseline_mae: float
    recent_mae: float

    @property
    def should_heal(self) -> bool:
        return self.status == "heal"


def _interval_coverage(
    actual: Sequence[float], intervals: Sequence[Tuple[float, float, float]]
) -> float:
    if not actual:
        return 1.0
    hits = sum(1 for a, (_, lo, hi) in zip(actual, intervals) if lo <= a <= hi)
    return hits / len(actual)


def confidence_decay(
    baseline_mae: float,
    recent_actual: Sequence[float],
    recent_pred: Sequence[float],
    recent_intervals: Sequence[Tuple[float, float, float]],
    thresholds: ConfidenceThresholds = ConfidenceThresholds(),
) -> ConfidenceReport:
    """Compute the confidence-decay score from recent performance vs. baseline."""
    n = min(len(recent_actual), len(recent_pred))
    if n == 0:
        return ConfidenceReport(0.0, "healthy", 0.0, 1.0, baseline_mae, baseline_mae)

    recent_mae = sum(abs(a - p) for a, p in zip(recent_actual, recent_pred)) / n

    # --- error-growth component -------------------------------------------------
    # Ratio of recent error to baseline error, squashed into [0, 1].
    denom = baseline_mae if baseline_mae > 1e-9 else 1e-9
    growth = (recent_mae - baseline_mae) / denom  # 0 == no growth, 1 == doubled
    error_component = max(0.0, min(1.0, growth))

    # --- coverage-breach component ---------------------------------------------
    coverage = _interval_coverage(recent_actual, recent_intervals)
    breach = max(0.0, thresholds.target_coverage - coverage)
    # normalize by target so a total coverage collapse maps to ~1
    coverage_component = min(1.0, breach / max(thresholds.target_coverage, 1e-9))

    # Weighted blend — error growth dominates, coverage breach amplifies.
    score = min(1.0, 0.65 * error_component + 0.35 * coverage_component)

    if score >= thresholds.heal:
        status = "heal"
    elif score >= thresholds.warn:
        status = "watch"
    else:
        status = "healthy"

    return ConfidenceReport(
        score=round(score, 4),
        status=status,
        error_growth=round(growth, 4),
        coverage=round(coverage, 4),
        baseline_mae=round(baseline_mae, 4),
        recent_mae=round(recent_mae, 4),
    )
