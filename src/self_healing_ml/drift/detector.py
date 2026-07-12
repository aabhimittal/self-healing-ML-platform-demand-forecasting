"""Statistical drift detection: PSI + a KS-style distributional distance.

These are the *quantitative* half of drift detection — they tell you **that** a
distribution moved. The semantic explainer (``drift/semantic.py``) supplies the
**why**. The two run together inside the self-healing orchestrator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

from ..config import DriftThresholds


@dataclass
class DriftReport:
    psi: float
    ks: float
    severity: str  # "none" | "warn" | "alert"
    reference_mean: float
    live_mean: float
    mean_shift_pct: float
    details: dict = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return self.severity == "alert"


def _bin_edges(reference: Sequence[float], bins: int) -> List[float]:
    lo, hi = min(reference), max(reference)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    step = (hi - lo) / bins
    edges = [lo + i * step for i in range(bins + 1)]
    edges[0] = -math.inf
    edges[-1] = math.inf
    return edges


def _histogram(data: Sequence[float], edges: Sequence[float]) -> List[float]:
    counts = [0] * (len(edges) - 1)
    for x in data:
        # linear scan is fine for the small windows used here
        for b in range(len(edges) - 1):
            if edges[b] <= x < edges[b + 1]:
                counts[b] += 1
                break
    total = sum(counts) or 1
    return [c / total for c in counts]


def population_stability_index(
    reference: Sequence[float], live: Sequence[float], bins: int = 10
) -> float:
    """PSI between a reference and a live sample. 0 == identical distributions."""
    if not reference or not live:
        return 0.0
    edges = _bin_edges(reference, bins)
    ref_p = _histogram(reference, edges)
    live_p = _histogram(live, edges)
    eps = 1e-6
    psi = 0.0
    for r, l in zip(ref_p, live_p):
        r = max(r, eps)
        l = max(l, eps)
        psi += (l - r) * math.log(l / r)
    return psi


def ks_distance(reference: Sequence[float], live: Sequence[float]) -> float:
    """Kolmogorov-Smirnov style statistic: max gap between empirical CDFs."""
    if not reference or not live:
        return 0.0
    grid = sorted(set(reference) | set(live))
    ref_sorted = sorted(reference)
    live_sorted = sorted(live)

    def cdf(sorted_data: List[float], x: float) -> float:
        # fraction of points <= x
        lo, hi = 0, len(sorted_data)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_data[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(sorted_data)

    return max(abs(cdf(ref_sorted, x) - cdf(live_sorted, x)) for x in grid)


def detect_drift(
    reference: Sequence[float],
    live: Sequence[float],
    thresholds: DriftThresholds = DriftThresholds(),
) -> DriftReport:
    """Combine PSI and KS into a single, thresholded drift report."""
    psi = population_stability_index(reference, live, thresholds.bins)
    ks = ks_distance(reference, live)

    ref_mean = sum(reference) / len(reference) if reference else 0.0
    live_mean = sum(live) / len(live) if live else 0.0
    shift_pct = 0.0 if ref_mean == 0 else 100.0 * (live_mean - ref_mean) / ref_mean

    if psi >= thresholds.psi_alert or ks >= thresholds.ks_alert:
        severity = "alert"
    elif psi >= thresholds.psi_warn:
        severity = "warn"
    else:
        severity = "none"

    return DriftReport(
        psi=round(psi, 4),
        ks=round(ks, 4),
        severity=severity,
        reference_mean=round(ref_mean, 3),
        live_mean=round(live_mean, 3),
        mean_shift_pct=round(shift_pct, 2),
        details={
            "psi_alert_threshold": thresholds.psi_alert,
            "ks_alert_threshold": thresholds.ks_alert,
        },
    )
