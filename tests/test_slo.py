import _bootstrap  # noqa: F401

from self_healing_ml.config import SLOConfig
from self_healing_ml.healing.slo import LatencySLO, percentile


def test_percentile_basic():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(data, 0.0) == 1
    assert percentile(data, 1.0) == 10
    assert percentile([], 0.5) == 0.0
    assert percentile([42], 0.9) == 42


def test_relax_on_sustained_slowdown():
    slo = LatencySLO(SLOConfig(initial_p95_ms=120.0))
    decision = slo.renegotiate([250.0] * 100)
    assert decision.action == "relax"
    assert decision.new_p95_ms > decision.old_p95_ms
    assert slo.p95_target_ms > 120.0


def test_relax_respects_ceiling():
    slo = LatencySLO(SLOConfig(initial_p95_ms=120.0, hard_ceiling_ms=200.0))
    decision = slo.renegotiate([900.0] * 100)
    assert decision.new_p95_ms <= 200.0


def test_tighten_when_comfortably_fast():
    slo = LatencySLO(SLOConfig(initial_p95_ms=200.0, floor_ms=40.0))
    decision = slo.renegotiate([60.0] * 100)
    assert decision.action == "tighten"
    assert decision.new_p95_ms < 200.0
    assert decision.new_p95_ms >= 40.0


def test_hold_within_band():
    slo = LatencySLO(SLOConfig(initial_p95_ms=120.0, renegotiation_band=0.15))
    decision = slo.renegotiate([118.0] * 100)
    assert decision.action == "hold"
    assert not decision.changed
    assert slo.p95_target_ms == 120.0
