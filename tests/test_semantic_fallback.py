import _bootstrap  # noqa: F401

import os

from self_healing_ml.drift.semantic import (
    explain_drift,
    build_signals,
    REMEDIATION,
    CAUSES,
)


class _Drift:
    psi = 0.4
    ks = 0.3
    severity = "alert"
    mean_shift_pct = 0.0


class _Conf:
    score = 0.7
    coverage = 0.4
    error_growth = 0.9


def _signals(**overrides):
    s = build_signals(_Drift(), _Conf(), live_promo_rate=0.0, live_min_throughput=1.0)
    s.update(overrides)
    return s


def test_fallback_used_without_key():
    d = explain_drift(_signals(), use_llm=False)
    assert d.source == "rule_based"
    assert d.cause in CAUSES
    assert d.recommended_action == REMEDIATION[d.cause]


def test_rule_based_detects_robotics_slowdown():
    d = explain_drift(_signals(live_min_throughput=0.6, mean_shift_pct=-40.0), use_llm=False)
    assert d.cause == "robotics_slowdown"
    assert d.recommended_action == "infra_action_renegotiate_slo"


def test_rule_based_detects_promotion():
    d = explain_drift(_signals(live_promo_rate=0.5, mean_shift_pct=25.0), use_llm=False)
    assert d.cause == "promotion"
    assert d.recommended_action == "widen_intervals_no_retrain"


def test_rule_based_detects_seasonality():
    d = explain_drift(_signals(ks=0.3, mean_shift_pct=2.0), use_llm=False)
    assert d.cause == "seasonality_shift"
    assert d.recommended_action == "retrain_on_recent"


def test_auto_mode_without_api_key_falls_back():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        d = explain_drift(_signals(), use_llm=None)
        assert d.source == "rule_based"
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_every_cause_has_remediation():
    for cause in CAUSES:
        assert cause in REMEDIATION
