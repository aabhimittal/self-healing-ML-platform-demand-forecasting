"""Auto-SLO renegotiation for inference latency.

A fixed latency SLO is brittle: it either pages on-call constantly when traffic
grows, or hides real regressions when set loosely. This module renegotiates the
p95 latency SLO from *observed* percentiles, within guard rails, so the platform
tightens the SLO when the service is comfortably fast and relaxes it (up to a
hard ceiling) when a genuine, sustained slowdown makes the current target
unrealistic — for example the robotics-slowdown scenario, where the fix is an
infra/SLO action rather than a model retrain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..config import SLOConfig


def percentile(data: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]); pure stdlib."""
    if not data:
        return 0.0
    xs = sorted(data)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


@dataclass
class SLODecision:
    old_p95_ms: float
    new_p95_ms: float
    observed_p95_ms: float
    observed_p99_ms: float
    action: str  # "hold" | "relax" | "tighten"
    reason: str

    @property
    def changed(self) -> bool:
        return self.action != "hold"


class LatencySLO:
    def __init__(self, config: SLOConfig = SLOConfig()) -> None:
        self.config = config
        self.p95_target_ms = config.initial_p95_ms

    def renegotiate(self, latencies_ms: Sequence[float]) -> SLODecision:
        cfg = self.config
        obs_p95 = percentile(latencies_ms, 0.95)
        obs_p99 = percentile(latencies_ms, 0.99)
        old = self.p95_target_ms
        band = cfg.renegotiation_band * old

        if obs_p95 > old + band:
            # sustained slowdown: relax toward observed p95, capped at the ceiling
            proposed = min(obs_p95, cfg.hard_ceiling_ms)
            if proposed > old:
                self.p95_target_ms = round(proposed, 2)
                return SLODecision(
                    old, self.p95_target_ms, round(obs_p95, 2), round(obs_p99, 2),
                    "relax",
                    f"observed p95 {obs_p95:.0f}ms exceeded SLO {old:.0f}ms by >"
                    f"{cfg.renegotiation_band:.0%}; relaxed to {self.p95_target_ms:.0f}ms "
                    f"(ceiling {cfg.hard_ceiling_ms:.0f}ms)",
                )
        elif obs_p95 < old - band:
            # comfortably fast: tighten toward observed p95, floored
            proposed = max(obs_p95, cfg.floor_ms)
            if proposed < old:
                self.p95_target_ms = round(proposed, 2)
                return SLODecision(
                    old, self.p95_target_ms, round(obs_p95, 2), round(obs_p99, 2),
                    "tighten",
                    f"observed p95 {obs_p95:.0f}ms comfortably under SLO {old:.0f}ms; "
                    f"tightened to {self.p95_target_ms:.0f}ms (floor {cfg.floor_ms:.0f}ms)",
                )

        return SLODecision(
            old, old, round(obs_p95, 2), round(obs_p99, 2), "hold",
            f"observed p95 {obs_p95:.0f}ms within {cfg.renegotiation_band:.0%} band of "
            f"SLO {old:.0f}ms",
        )
