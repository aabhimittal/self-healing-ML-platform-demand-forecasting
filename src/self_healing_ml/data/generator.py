"""Synthetic retail demand-series generator with injectable drift events.

The generator produces a daily demand signal built from a trend, weekly and
yearly seasonality, and noise. On top of that it can inject the three drift
scenarios the platform is designed to *explain*:

    - ``promotion``          : a sudden multiplicative demand spike (e.g. a sale)
    - ``seasonality_shift``  : a phase/amplitude change in the weekly pattern
    - ``robotics_slowdown``  : a fulfilment-capacity ceiling that clips demand,
                               modelling a warehouse-robotics throughput drop

Everything is pure-stdlib (``math`` + ``random``) so the platform has zero
runtime dependencies and the demo runs anywhere.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DriftEvent:
    """A drift scenario injected over a day range ``[start, end)``.

    ``kind`` is one of ``promotion``, ``seasonality_shift``, ``robotics_slowdown``.
    ``magnitude`` scales the effect (interpretation depends on ``kind``).
    """

    kind: str
    start: int
    end: int
    magnitude: float = 1.0

    def active(self, day: int) -> bool:
        return self.start <= day < self.end


@dataclass
class DemandSeries:
    """A generated demand series plus the exogenous signals a real platform sees."""

    values: List[float]
    days: List[int] = field(default_factory=list)
    day_of_week: List[int] = field(default_factory=list)
    # Exogenous features an ops team would actually have on hand:
    promo_flag: List[int] = field(default_factory=list)
    robotics_throughput: List[float] = field(default_factory=list)
    events: List[DriftEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    def window(self, start: int, end: int) -> List[float]:
        return self.values[start:end]

    def active_events(self, day: int) -> List[DriftEvent]:
        return [e for e in self.events if e.active(day)]


_VALID_KINDS = {"promotion", "seasonality_shift", "robotics_slowdown"}


def generate_demand(
    n_days: int = 180,
    base: float = 100.0,
    trend: float = 0.05,
    weekly_amplitude: float = 20.0,
    yearly_amplitude: float = 15.0,
    noise: float = 4.0,
    events: Optional[List[DriftEvent]] = None,
    seed: Optional[int] = 7,
) -> DemandSeries:
    """Generate a daily demand series with optional injected drift events."""
    events = events or []
    for e in events:
        if e.kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown drift kind {e.kind!r}; expected one of {sorted(_VALID_KINDS)}"
            )
    rng = random.Random(seed)

    values: List[float] = []
    days: List[int] = []
    dow: List[int] = []
    promo_flag: List[int] = []
    throughput: List[float] = []

    for day in range(n_days):
        active = [e for e in events if e.active(day)]

        # --- baseline structure ------------------------------------------------
        weekly_phase = 0.0
        weekly_amp = weekly_amplitude
        for e in active:
            if e.kind == "seasonality_shift":
                # shift the weekly peak and stretch its amplitude
                weekly_phase += math.pi * 0.5 * e.magnitude
                weekly_amp *= 1.0 + 0.5 * e.magnitude

        weekly = weekly_amp * math.sin(2 * math.pi * (day % 7) / 7 + weekly_phase)
        yearly = yearly_amplitude * math.sin(2 * math.pi * day / 365.0)
        level = base + trend * day + weekly + yearly
        level += rng.gauss(0, noise)

        # --- promotion: multiplicative spike ----------------------------------
        promo_on = 0
        for e in active:
            if e.kind == "promotion":
                level *= 1.0 + 0.6 * e.magnitude
                promo_on = 1

        # --- robotics slowdown: fulfilment capacity ceiling clips demand ------
        capacity = math.inf
        for e in active:
            if e.kind == "robotics_slowdown":
                # throughput drops -> effective ceiling below normal demand
                capacity = min(capacity, level * (1.0 - 0.4 * e.magnitude))
        realized = min(level, capacity)
        realized = max(realized, 0.0)

        # normalized robotics throughput signal (1.0 == healthy)
        tput = 1.0
        for e in active:
            if e.kind == "robotics_slowdown":
                tput = min(tput, 1.0 - 0.4 * e.magnitude)

        values.append(round(realized, 3))
        days.append(day)
        dow.append(day % 7)
        promo_flag.append(promo_on)
        throughput.append(round(tput, 3))

    return DemandSeries(
        values=values,
        days=days,
        day_of_week=dow,
        promo_flag=promo_flag,
        robotics_throughput=throughput,
        events=list(events),
    )
