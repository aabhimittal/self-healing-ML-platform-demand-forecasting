import _bootstrap  # noqa: F401

import pytest

from self_healing_ml.data.generator import generate_demand
from self_healing_ml.models.forecaster import (
    DemandForecaster,
    mean_absolute_error,
    mape,
)


def _fit_on(series):
    m = DemandForecaster()
    return m.fit(series.values, series.day_of_week, series.promo_flag, series.robotics_throughput)


def test_fit_predict_aligns_and_learns():
    s = generate_demand(n_days=120, seed=2, noise=2.0)
    m = _fit_on(s)
    preds = m.predict(s.values, s.day_of_week, s.promo_flag, s.robotics_throughput)
    assert len(preds) == len(s) - 7
    actual = s.values[7:]
    mae = mean_absolute_error(actual, preds)
    # a fitted linear model should comfortably beat a naive persistence baseline
    naive = [s.values[i - 1] for i in range(7, len(s))]
    assert mae < mean_absolute_error(actual, naive)


def test_prediction_interval_brackets_point():
    s = generate_demand(n_days=90, seed=4)
    m = _fit_on(s)
    intervals = m.predict_interval(s.values, s.day_of_week, s.promo_flag, s.robotics_throughput)
    for point, lo, hi in intervals:
        assert lo <= point <= hi
        assert hi - lo > 0


def test_interval_coverage_is_reasonable_on_stationary_data():
    s = generate_demand(n_days=150, seed=9, trend=0.0, yearly_amplitude=4.0, noise=3.0)
    m = _fit_on(s)
    intervals = m.predict_interval(s.values, s.day_of_week, s.promo_flag, s.robotics_throughput)
    actual = s.values[7:]
    hits = sum(1 for a, (_, lo, hi) in zip(actual, intervals) if lo <= a <= hi)
    coverage = hits / len(actual)
    # a ~90% interval on in-sample residuals should cover most points
    assert coverage >= 0.8


def test_fit_requires_minimum_history():
    s = generate_demand(n_days=6, seed=1)
    with pytest.raises(ValueError):
        _fit_on(s)


def test_metrics_helpers():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0
    assert mean_absolute_error([1, 2], [2, 4]) == pytest.approx(1.5)
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert mape([], []) == 0.0
