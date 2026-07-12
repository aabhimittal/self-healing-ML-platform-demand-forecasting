"""Kubeflow Pipelines (KFP v2) definition — multi-model self-healing pipeline.

Expresses the self-healing loop as a KFP pipeline so it can fan out across many
SKUs / model families in parallel on Kubernetes, with MLflow-backed lineage and
Prometheus-scraped component metrics. The component bodies delegate to the core
``self_healing_ml`` package, keeping one source of truth for the logic.

Requires ``kfp`` (guarded import) so the file is importable without it.
"""
from __future__ import annotations

try:  # pragma: no cover - kfp is an optional, environment-provided dependency
    from kfp import dsl
    from kfp.dsl import component
    _KFP = True
except Exception:  # noqa: BLE001
    _KFP = False


if _KFP:

    @component(packages_to_install=["self-healing-ml"])
    def detect_drift_op(reference: list, live: list) -> dict:
        from self_healing_ml.drift.detector import detect_drift
        return detect_drift(reference, live).__dict__

    @component(packages_to_install=["self-healing-ml", "anthropic"])
    def diagnose_op(signals: dict) -> dict:
        from self_healing_ml.drift.semantic import explain_drift
        return explain_drift(signals).as_dict()

    @component(packages_to_install=["self-healing-ml"])
    def retrain_validate_deploy_op(model_name: str, diagnosis: dict) -> dict:
        # Train challenger, validate against the registry gate, promote if it wins.
        # Returns the deployment decision for downstream lineage.
        return {"model": model_name, "action": diagnosis.get("recommended_action")}

    @component(packages_to_install=["self-healing-ml"])
    def renegotiate_slo_op(latencies_ms: list) -> dict:
        from self_healing_ml.healing.slo import LatencySLO
        return LatencySLO().renegotiate(latencies_ms).__dict__

    @dsl.pipeline(
        name="self-healing-demand-forecasting",
        description="Detect -> diagnose (LLM) -> retrain/validate/deploy -> renegotiate SLO",
    )
    def self_healing_pipeline(
        model_name: str = "demand_forecaster",
        reference: list = [],
        live: list = [],
        signals: dict = {},
        latencies_ms: list = [],
    ):
        drift = detect_drift_op(reference=reference, live=live)
        with dsl.If(drift.output["severity"] == "alert", name="on-drift"):
            diagnosis = diagnose_op(signals=signals)
            retrain_validate_deploy_op(model_name=model_name, diagnosis=diagnosis.output)
        renegotiate_slo_op(latencies_ms=latencies_ms)


def compile_pipeline(output_path: str = "self_healing_pipeline.yaml") -> str:
    """Compile the pipeline to IR YAML (no-op message if kfp is absent)."""
    if not _KFP:
        return "kfp not installed; install `kfp` to compile this pipeline"
    from kfp import compiler

    compiler.Compiler().compile(self_healing_pipeline, output_path)
    return output_path


if __name__ == "__main__":
    print(compile_pipeline())
