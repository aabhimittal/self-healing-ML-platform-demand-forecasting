"""A tiny Prometheus-style metrics registry.

Supports counters, gauges, and histograms and renders them in the Prometheus
text-exposition format so the same code that drives the demo could be scraped by
a real Prometheus server. The self-healing orchestrator emits drift, latency,
retrain, and rollback metrics through this registry; the Grafana dashboard in
``observability/grafana_dashboard.json`` visualizes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Default histogram buckets tuned for inference latency in milliseconds.
DEFAULT_BUCKETS = (25, 50, 75, 100, 150, 200, 300, 500, 1000)


def _label_key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _fmt_labels(labels: Tuple[Tuple[str, str], ...], extra: Tuple[Tuple[str, str], ...] = ()) -> str:
    items = list(labels) + list(extra)
    if not items:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in items)
    return "{" + inner + "}"


@dataclass
class _Series:
    kind: str  # counter | gauge | histogram
    help: str
    values: Dict[Tuple[Tuple[str, str], ...], float] = field(default_factory=dict)
    # histogram-only:
    buckets: Tuple[float, ...] = ()
    bucket_counts: Dict[Tuple[Tuple[str, str], ...], List[int]] = field(default_factory=dict)
    sums: Dict[Tuple[Tuple[str, str], ...], float] = field(default_factory=dict)
    counts: Dict[Tuple[Tuple[str, str], ...], int] = field(default_factory=dict)


class MetricsRegistry:
    def __init__(self) -> None:
        self._series: Dict[str, _Series] = {}

    def _ensure(self, name: str, kind: str, help: str, buckets: Tuple[float, ...] = ()) -> _Series:
        s = self._series.get(name)
        if s is None:
            s = _Series(kind=kind, help=help, buckets=buckets)
            self._series[name] = s
        elif s.kind != kind:
            raise ValueError(f"metric {name!r} already registered as {s.kind}, not {kind}")
        return s

    # --- counter ------------------------------------------------------------
    def inc(self, name: str, amount: float = 1.0, help: str = "", **labels: str) -> None:
        s = self._ensure(name, "counter", help)
        key = _label_key(labels)
        s.values[key] = s.values.get(key, 0.0) + amount

    # --- gauge --------------------------------------------------------------
    def set(self, name: str, value: float, help: str = "", **labels: str) -> None:
        s = self._ensure(name, "gauge", help)
        s.values[_label_key(labels)] = value

    # --- histogram ----------------------------------------------------------
    def observe(
        self,
        name: str,
        value: float,
        help: str = "",
        buckets: Tuple[float, ...] = DEFAULT_BUCKETS,
        **labels: str,
    ) -> None:
        s = self._ensure(name, "histogram", help, buckets)
        key = _label_key(labels)
        counts = s.bucket_counts.setdefault(key, [0] * len(s.buckets))
        for i, b in enumerate(s.buckets):
            if value <= b:
                counts[i] += 1
        s.sums[key] = s.sums.get(key, 0.0) + value
        s.counts[key] = s.counts.get(key, 0) + 1

    # --- read ---------------------------------------------------------------
    def value(self, name: str, **labels: str) -> float:
        s = self._series[name]
        return s.values.get(_label_key(labels), 0.0)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Flat dict view, useful for tests and the demo timeline."""
        out: Dict[str, Dict[str, float]] = {}
        for name, s in self._series.items():
            if s.kind in ("counter", "gauge"):
                out[name] = {_fmt_labels(k): v for k, v in s.values.items()}
            else:
                out[name] = {
                    _fmt_labels(k): s.counts.get(k, 0) for k in s.bucket_counts
                }
        return out

    # --- prometheus text exposition ----------------------------------------
    def render(self) -> str:
        lines: List[str] = []
        for name, s in self._series.items():
            if s.help:
                lines.append(f"# HELP {name} {s.help}")
            lines.append(f"# TYPE {name} {s.kind}")
            if s.kind in ("counter", "gauge"):
                for key, v in s.values.items():
                    lines.append(f"{name}{_fmt_labels(key)} {v}")
            else:  # histogram
                # bucket_counts already stores cumulative counts (each observation
                # increments every bucket whose upper bound it falls under), so the
                # stored value is exactly the Prometheus le=<b> series.
                for key, counts in s.bucket_counts.items():
                    for b, c in zip(s.buckets, counts):
                        le = (("le", str(b)),)
                        lines.append(f"{name}_bucket{_fmt_labels(key, le)} {c}")
                    inf = (("le", "+Inf"),)
                    lines.append(
                        f"{name}_bucket{_fmt_labels(key, inf)} {s.counts.get(key, 0)}"
                    )
                    lines.append(f"{name}_sum{_fmt_labels(key)} {s.sums.get(key, 0.0)}")
                    lines.append(f"{name}_count{_fmt_labels(key)} {s.counts.get(key, 0)}")
        return "\n".join(lines) + "\n"
