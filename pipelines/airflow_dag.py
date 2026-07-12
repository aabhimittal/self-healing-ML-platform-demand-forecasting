"""Airflow DAG — scheduled batch retraining for the self-healing platform.

This DAG runs one self-healing cycle on a daily schedule. Each task maps to a
step in ``SelfHealingOrchestrator.run_cycle`` so the same control loop that the
demo runs in-process is here expressed as an orchestrated, observable pipeline:

    detect_drift -> diagnose (LLM) -> [retrain -> validate -> deploy] -> renegotiate_slo

The task callables import the core package, so there is a single source of truth
for the logic — Airflow only supplies scheduling, retries, and lineage.

Requires ``apache-airflow`` (guarded import) so this file is safe to keep in a
repo that is also used without Airflow installed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

try:  # pragma: no cover - Airflow is an optional, environment-provided dependency
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    _AIRFLOW = True
except Exception:  # noqa: BLE001
    _AIRFLOW = False


DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


# --- task callables (import the core package; XComs carry state) -------------
def detect_drift_task(**context):
    """Compute PSI/KS + confidence decay; push a drift verdict to XCom."""
    # In production this reads the feature store; here it is illustrative.
    from self_healing_ml.drift.detector import detect_drift
    reference = context["params"]["reference_window"]
    live = context["params"]["live_window"]
    report = detect_drift(reference, live)
    context["ti"].xcom_push(key="drift", value=report.__dict__)
    return report.severity


def branch_on_drift(**context):
    severity = context["ti"].xcom_pull(key="drift")["severity"]
    return "diagnose_and_retrain" if severity == "alert" else "renegotiate_slo"


def diagnose_and_retrain_task(**context):
    """LLM semantic diagnosis + champion/challenger retrain + gated deploy."""
    from self_healing_ml.drift.semantic import explain_drift
    signals = context["ti"].xcom_pull(key="drift")
    diagnosis = explain_drift(signals)
    context["ti"].xcom_push(key="diagnosis", value=diagnosis.as_dict())
    # a real deployment would train, validate against the registry gate, and
    # promote via the model registry here.
    return diagnosis.recommended_action


def renegotiate_slo_task(**context):
    """Auto-renegotiate the inference latency SLO from observed percentiles."""
    from self_healing_ml.healing.slo import LatencySLO
    latencies = context["params"]["latencies_ms"]
    decision = LatencySLO().renegotiate(latencies)
    context["ti"].xcom_push(key="slo", value=decision.__dict__)
    return decision.action


if _AIRFLOW:
    with DAG(
        dag_id="self_healing_demand_forecast",
        description="Daily self-healing cycle for demand forecasting models",
        default_args=DEFAULT_ARGS,
        schedule="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "self-healing", "forecasting"],
        params={"reference_window": [], "live_window": [], "latencies_ms": []},
    ) as dag:
        detect = PythonOperator(task_id="detect_drift", python_callable=detect_drift_task)
        branch = BranchPythonOperator(task_id="branch_on_drift", python_callable=branch_on_drift)
        diagnose = PythonOperator(
            task_id="diagnose_and_retrain", python_callable=diagnose_and_retrain_task
        )
        renegotiate = PythonOperator(
            task_id="renegotiate_slo",
            python_callable=renegotiate_slo_task,
            trigger_rule="none_failed_min_one_success",
        )

        detect >> branch >> [diagnose, renegotiate]
        diagnose >> renegotiate
