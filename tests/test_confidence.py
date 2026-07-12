import _bootstrap  # noqa: F401

from self_healing_ml.drift.confidence import confidence_decay


def _intervals(points, margin=5.0):
    return [(p, p - margin, p + margin) for p in points]


def test_healthy_model_scores_low():
    actual = [100.0, 101.0, 99.0, 100.0, 102.0]
    pred = [100.0, 100.5, 99.5, 100.0, 101.5]
    rep = confidence_decay(1.0, actual, pred, _intervals(pred))
    assert rep.status == "healthy"
    assert rep.score < 0.35
    assert not rep.should_heal


def test_error_growth_drives_decay():
    actual = [100.0] * 6
    pred = [130.0] * 6  # large, persistent error vs baseline of 1.0
    rep = confidence_decay(1.0, actual, pred, _intervals(pred, margin=1.0))
    assert rep.should_heal
    assert rep.status == "heal"
    assert rep.score >= 0.55


def test_coverage_breach_contributes():
    actual = [100.0, 100.0, 100.0, 100.0]
    pred = [100.0, 100.0, 100.0, 100.0]
    # intervals that never contain the actuals -> full coverage breach
    bad = [(100.0, 200.0, 300.0) for _ in actual]
    rep = confidence_decay(1.0, actual, pred, bad)
    assert rep.coverage == 0.0
    assert rep.score > 0.0


def test_empty_recent_window_is_healthy():
    rep = confidence_decay(2.0, [], [], [])
    assert rep.status == "healthy"
    assert rep.score == 0.0
