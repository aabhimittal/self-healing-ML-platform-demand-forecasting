import _bootstrap  # noqa: F401

import pytest

from self_healing_ml.observability.metrics import MetricsRegistry


def test_counter_accumulates():
    m = MetricsRegistry()
    m.inc("hits_total")
    m.inc("hits_total", 4)
    assert m.value("hits_total") == 5.0


def test_gauge_sets_latest():
    m = MetricsRegistry()
    m.set("temp", 1.0)
    m.set("temp", 9.0)
    assert m.value("temp") == 9.0


def test_labels_are_independent_series():
    m = MetricsRegistry()
    m.inc("diagnoses_total", cause="promotion")
    m.inc("diagnoses_total", cause="promotion")
    m.inc("diagnoses_total", cause="robotics_slowdown")
    assert m.value("diagnoses_total", cause="promotion") == 2.0
    assert m.value("diagnoses_total", cause="robotics_slowdown") == 1.0


def test_histogram_buckets_are_cumulative_and_monotonic():
    m = MetricsRegistry()
    for v in (10, 30, 60, 120, 400):
        m.observe("lat_ms", v, buckets=(25, 50, 100, 200, 500))
    text = m.render()
    lines = [l for l in text.splitlines() if l.startswith("lat_ms_bucket")]
    counts = [int(l.rsplit(" ", 1)[1]) for l in lines]
    # cumulative buckets must be non-decreasing and end at the total count (5)
    assert counts == sorted(counts)
    assert counts[-1] == 5


def test_kind_conflict_raises():
    m = MetricsRegistry()
    m.inc("x")
    with pytest.raises(ValueError):
        m.set("x", 1.0)


def test_render_is_valid_prometheus_text():
    m = MetricsRegistry()
    m.set("g", 2.0, help="a gauge")
    text = m.render()
    assert "# TYPE g gauge" in text
    assert "# HELP g a gauge" in text
    assert text.endswith("\n")
