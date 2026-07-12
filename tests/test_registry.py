import _bootstrap  # noqa: F401

from self_healing_ml.registry.model_registry import ModelRegistry


def test_register_and_promote():
    reg = ModelRegistry()
    v1 = reg.register("model-a", metrics={"mae": 5.0})
    assert v1.version == 1
    assert v1.stage == "none"
    reg.promote(v1.version)
    assert reg.production().version == 1


def test_promotion_archives_incumbent():
    reg = ModelRegistry()
    v1 = reg.register("m1", metrics={"mae": 5.0})
    reg.promote(v1.version)
    v2 = reg.register("m2", metrics={"mae": 3.0}, parent_version=v1.version)
    reg.promote(v2.version)
    assert reg.production().version == 2
    assert reg.get(1).stage == "archived"


def test_rollback_restores_previous_production():
    reg = ModelRegistry()
    v1 = reg.register("m1", metrics={"mae": 5.0})
    reg.promote(v1.version)
    v2 = reg.register("m2", metrics={"mae": 3.0}, parent_version=v1.version)
    reg.promote(v2.version)
    restored = reg.rollback()
    assert restored is not None
    assert restored.version == 1
    assert reg.production().version == 1
    assert reg.get(2).stage == "archived"


def test_rollback_with_nothing_to_restore():
    reg = ModelRegistry()
    v1 = reg.register("m1")
    reg.promote(v1.version)
    assert reg.rollback() is None


def test_lineage_chain():
    reg = ModelRegistry()
    v1 = reg.register("m1")
    v2 = reg.register("m2", parent_version=v1.version)
    v3 = reg.register("m3", parent_version=v2.version)
    assert reg.lineage(v3.version) == [1, 2, 3]
    assert reg.lineage(v1.version) == [1]


def test_transition_history_recorded():
    reg = ModelRegistry()
    v1 = reg.register("m1")
    reg.stage(v1.version)
    reg.promote(v1.version)
    kinds = [(t.from_stage, t.to_stage) for t in reg.history]
    assert ("none", "staging") in kinds
    assert ("staging", "production") in kinds
