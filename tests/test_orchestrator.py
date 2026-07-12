import _bootstrap  # noqa: F401

from self_healing_ml.config import DEFAULT_CONFIG
from self_healing_ml.data.generator import DriftEvent, generate_demand
from self_healing_ml.healing.orchestrator import SelfHealingOrchestrator
from self_healing_ml.scenario import run_scenario


def _series():
    events = [
        DriftEvent("seasonality_shift", 120, 150, 1.0),
        DriftEvent("promotion", 150, 165, 1.0),
        DriftEvent("robotics_slowdown", 180, 210, 1.0),
    ]
    return generate_demand(
        n_days=210, trend=0.02, yearly_amplitude=8.0, events=events, seed=7
    )


def test_bootstrap_promotes_initial_model():
    orch = SelfHealingOrchestrator(use_llm=False)
    mv = orch.bootstrap(_series(), train_end=90)
    assert orch.registry.production().version == mv.version
    assert orch.metrics.value("shml_baseline_mae") > 0


def test_healthy_cycle_takes_no_action():
    orch = SelfHealingOrchestrator(use_llm=False)
    s = _series()
    orch.bootstrap(s, train_end=90)
    rep = orch.run_cycle(s, 90, 120, [78.0] * 50)
    assert rep.action == "monitor"
    assert not rep.healed
    assert rep.production_version == 1


def test_seasonality_cycle_retrains_and_promotes():
    orch = SelfHealingOrchestrator(use_llm=False)
    s = _series()
    orch.bootstrap(s, train_end=90)
    orch.run_cycle(s, 90, 120, [78.0] * 50)  # healthy
    rep = orch.run_cycle(s, 120, 150, [80.0] * 50)  # seasonality
    assert rep.diagnosis is not None
    assert rep.diagnosis.cause == "seasonality_shift"
    assert rep.healed
    assert rep.production_version == 2


def test_promotion_cycle_does_not_retrain():
    orch = SelfHealingOrchestrator(use_llm=False)
    s = _series()
    orch.bootstrap(s, train_end=90)
    orch.run_cycle(s, 90, 120, [78.0] * 50)
    orch.run_cycle(s, 120, 150, [80.0] * 50)
    before = orch.registry.production().version
    rep = orch.run_cycle(s, 150, 180, [82.0] * 50)
    assert rep.diagnosis.cause == "promotion"
    assert rep.action == "widen_intervals_no_retrain"
    assert rep.production_version == before  # no new promotion


def test_robotics_cycle_renegotiates_slo():
    orch = SelfHealingOrchestrator(use_llm=False)
    s = _series()
    orch.bootstrap(s, train_end=90)
    for start, end in [(90, 120), (120, 150), (150, 180)]:
        orch.run_cycle(s, start, end, [80.0] * 50)
    rep = orch.run_cycle(s, 180, 210, [235.0] * 50)
    assert rep.diagnosis.cause == "robotics_slowdown"
    assert rep.slo.action == "relax"
    assert orch.slo.p95_target_ms > DEFAULT_CONFIG.slo.initial_p95_ms


def test_rollback_reverts_production():
    orch = SelfHealingOrchestrator(use_llm=False)
    s = _series()
    orch.bootstrap(s, train_end=90)
    orch.run_cycle(s, 90, 120, [78.0] * 50)
    orch.run_cycle(s, 120, 150, [80.0] * 50)  # promotes v2
    assert orch.registry.production().version == 2
    restored = orch.rollback()
    assert restored.version == 1
    assert orch.registry.production().version == 1


def test_full_scenario_end_to_end():
    result = run_scenario(seed=7, use_llm=False, verbose=False)
    assert len(result.cycles) == 4
    # at least one heal (seasonality) and a rollback target
    assert any(c.healed for c in result.cycles)
    assert result.rolled_back_to == 1
    causes = {c.diagnosis.cause for c in result.cycles if c.diagnosis}
    assert {"seasonality_shift", "promotion", "robotics_slowdown"} <= causes


def test_audit_trail_is_populated():
    result = run_scenario(seed=7, use_llm=False, verbose=False)
    kinds = {e.kind for e in result.orchestrator.audit}
    assert "bootstrap" in kinds
    assert "diagnosis" in kinds
    assert "rollback" in kinds
