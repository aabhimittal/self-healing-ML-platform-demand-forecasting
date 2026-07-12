import _bootstrap  # noqa: F401

from self_healing_ml.config import DriftThresholds
from self_healing_ml.drift.detector import (
    detect_drift,
    population_stability_index,
    ks_distance,
)


def test_psi_zero_for_identical_distributions():
    data = [float(i % 10) for i in range(100)]
    assert population_stability_index(data, data) < 1e-6


def test_psi_grows_with_shift():
    ref = [float(i % 10) for i in range(200)]
    live = [float(i % 10) + 8 for i in range(200)]
    assert population_stability_index(ref, live) > 0.25


def test_ks_distance_bounds():
    ref = list(range(100))
    assert ks_distance(ref, ref) == 0.0
    shifted = [x + 1000 for x in ref]
    assert ks_distance(ref, shifted) == 1.0


def test_detect_drift_severity_levels():
    ref = [10.0 + (i % 5) for i in range(120)]
    # near-identical -> none
    quiet = detect_drift(ref, [10.0 + (i % 5) for i in range(60)])
    assert quiet.severity in ("none", "warn")
    assert not quiet.drifted
    # large shift -> alert
    loud = detect_drift(ref, [80.0 + (i % 5) for i in range(60)])
    assert loud.severity == "alert"
    assert loud.drifted
    assert loud.mean_shift_pct > 0


def test_custom_thresholds_respected():
    ref = [1.0, 2.0, 3.0, 4.0] * 30
    live = [2.0, 3.0, 4.0, 5.0] * 30
    strict = detect_drift(ref, live, DriftThresholds(psi_alert=0.01, ks_alert=0.01))
    assert strict.severity == "alert"


def test_empty_inputs_are_safe():
    r = detect_drift([], [])
    assert r.psi == 0.0
    assert r.severity == "none"
