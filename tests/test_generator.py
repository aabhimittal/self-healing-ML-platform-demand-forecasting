import _bootstrap  # noqa: F401

import pytest

from self_healing_ml.data.generator import DriftEvent, generate_demand


def test_generate_shape_and_signals():
    s = generate_demand(n_days=60, seed=1)
    assert len(s) == 60
    assert len(s.day_of_week) == 60
    assert len(s.promo_flag) == 60
    assert len(s.robotics_throughput) == 60
    assert all(v >= 0 for v in s.values)
    assert set(s.day_of_week) <= set(range(7))


def test_generation_is_deterministic_with_seed():
    a = generate_demand(n_days=40, seed=42)
    b = generate_demand(n_days=40, seed=42)
    assert a.values == b.values


def test_promotion_raises_mean_and_sets_flag():
    base = generate_demand(n_days=60, seed=3)
    promo = generate_demand(
        n_days=60,
        seed=3,
        events=[DriftEvent("promotion", 30, 45, magnitude=1.0)],
    )
    base_win = sum(base.values[30:45]) / 15
    promo_win = sum(promo.values[30:45]) / 15
    assert promo_win > base_win
    assert sum(promo.promo_flag[30:45]) == 15
    assert sum(promo.promo_flag[:30]) == 0


def test_robotics_slowdown_clips_demand_and_throughput():
    base = generate_demand(n_days=60, seed=5)
    slow = generate_demand(
        n_days=60,
        seed=5,
        events=[DriftEvent("robotics_slowdown", 30, 45, magnitude=1.0)],
    )
    assert sum(slow.values[30:45]) < sum(base.values[30:45])
    assert min(slow.robotics_throughput[30:45]) < 0.85
    assert min(base.robotics_throughput) == 1.0


def test_unknown_drift_kind_rejected():
    with pytest.raises(ValueError):
        generate_demand(n_days=20, events=[DriftEvent("meteor_strike", 5, 10)])


def test_active_events_query():
    ev = DriftEvent("promotion", 10, 20)
    s = generate_demand(n_days=30, events=[ev], seed=1)
    assert s.active_events(15) == [ev]
    assert s.active_events(25) == []
