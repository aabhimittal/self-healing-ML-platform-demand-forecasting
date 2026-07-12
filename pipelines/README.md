# Stack Integration

These manifests show how the in-process self-healing loop
(`self_healing_ml.healing.orchestrator.SelfHealingOrchestrator`) maps onto the
production MLOps stack. Each one delegates to the **same core package**, so the
logic exercised by `examples/run_demo.py` is the logic that runs in production —
the orchestrators only add scheduling, distribution, retries, and lineage.

| Manifest | Stack component | Role |
|---|---|---|
| `airflow_dag.py` | **Airflow** | Daily batch retraining schedule; one DAG run == one `run_cycle`. Branches on the drift verdict. |
| `kubeflow_pipeline.py` | **Kubeflow Pipelines** | Fans the loop out across many SKUs / model families on Kubernetes, with per-component metrics. |
| `step_functions.json` | **AWS Step Functions** | The retrain → validate → deploy → canary → **rollback** state machine, including the LLM diagnosis branch. |
| `../observability/prometheus_rules.yml` | **Prometheus** | Alerting rules over the metrics emitted by `observability/metrics.py`. |
| `../observability/grafana_dashboard.json` | **Grafana** | Drift, confidence-decay, latency-SLO, and heal/rollback panels. |

## Loop → manifest mapping

```
orchestrator.run_cycle()          Airflow task        Step Functions state
──────────────────────────────    ────────────────    ────────────────────
detect_drift + confidence_decay   detect_drift        DetectDrift / IsActionable
explain_drift (LLM)               diagnose_and_retrain DiagnoseWithLLM / RouteByCause
retrain + champion/challenger     diagnose_and_retrain Retrain / Validate / ValidationGate
promote / rollback                (registry)          Deploy / CanaryCheck / Rollback
LatencySLO.renegotiate            renegotiate_slo     RenegotiateSLO
```

## Model lineage & metrics

- **MLflow** — `registry/model_registry.py` mirrors the MLflow model-registry
  concepts (versions, stages `staging`/`production`/`archived`, transitions,
  lineage). Swap the in-memory store for the MLflow tracking + registry client
  and the interface is unchanged.
- **Prometheus/Grafana** — the orchestrator emits `shml_*` metrics through
  `observability/metrics.py`, which renders Prometheus text exposition
  (`python -m self_healing_ml.cli metrics`).
- **Datadog** — infra-cost anomaly detection would consume the same latency /
  throughput signals that drive `LatencySLO`; the robotics-slowdown scenario is
  exactly the cost-anomaly case where the remediation is an infra/SLO action
  rather than a model retrain.

None of these files are imported by the core package or the test suite; the
Airflow/KFP imports are guarded so the repo stays usable without those
frameworks installed.
