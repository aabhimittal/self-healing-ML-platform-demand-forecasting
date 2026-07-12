# Self-Healing ML Platform for Demand Forecasting

**LLMOps + AIOps** — an ML platform where demand-forecasting models **diagnose their
own degradation, retrain, redeploy, and tune their own infra**. The novel twist:
instead of a human noticing drift on a dashboard, an **LLM agent explains *why* drift
happened** — a promotion, a seasonality shift, or a robotics/fulfilment slowdown — and
that *cause* decides the remediation.

```
detect ──▶ diagnose (LLM) ──▶ decide ──▶ act ──▶ (rollback in one call)
 │            │                  │          │
 │            │                  │          ├─ retrain → validate → deploy
 │            │                  │          ├─ widen prediction intervals (no retrain)
 │            │                  │          └─ renegotiate the latency SLO
 │            │                  └─ remediation is chosen from the diagnosed cause
 │            └─ Claude reads the drift signals + business context
 └─ statistical drift (PSI/KS) + model confidence-decay score
```

> The core is **pure Python standard library** — no numpy, pandas, or ML frameworks —
> so the whole platform, its demo, and its tests run anywhere with **zero install**.
> The LLM diagnosis uses the official `anthropic` SDK with **Claude Opus 4.8** when a
> key is present, and **falls back to a deterministic rule-based explainer** otherwise.

---

## Quickstart

```bash
# zero-install entry points (no dependencies, no packaging needed)
python examples/run_demo.py            # narrated detect → diagnose → heal → rollback timeline
python tests/run_tests.py               # run the 52-test suite with zero installs

# the CLI needs the package importable — install it (editable) or set PYTHONPATH
pip install -e .                        # then:
python -m self_healing_ml.cli demo      # same scenario, via the CLI
python -m self_healing_ml.cli metrics   # print the Prometheus text exposition
#   …or without installing:  PYTHONPATH=src python -m self_healing_ml.cli demo

# optional: use Claude for the semantic diagnosis
pip install -r requirements-llm.txt
export ANTHROPIC_API_KEY=...            # or `ant auth login`
python examples/run_demo.py --llm
```

### What the demo does

One demand series is walked through four monitoring cycles that exercise **every**
remediation path, then a one-call rollback:

| cycle | scenario | detected | diagnosed cause | action |
|---|---|---|---|---|
| 1 | healthy traffic | confidence healthy | — | **monitor** (no wasteful retrain) |
| 2 | seasonality shift | drift alert + confidence heal | `seasonality_shift` | **retrain → validate → promote** |
| 3 | promotion spike | drift alert + confidence heal | `promotion` | **widen intervals, no retrain** |
| 4 | robotics slowdown | drift alert + latency breach | `robotics_slowdown` | **renegotiate latency SLO** |
| — | rollback | — | — | production reverts **v2 → v1** in one call |

The key design decision: **the platform only retrains when the model's own reliability
has decayed** — not on cosmetic input drift. A promotion is a self-resolving spike (widen
intervals, don't retrain on it); a robotics slowdown is a *supply-side* capacity ceiling,
so the fix is an infra/SLO action, not a model change. Getting the *cause* right is what
avoids needless retrains and regressions.

---

## The three innovations

### 1. LLM-based semantic drift detection (`drift/semantic.py`)
Statistical detectors tell you *that* a distribution moved; they can't tell you *why*,
and "why" determines the correct fix. `explain_drift()` hands Claude the quantitative
drift signals plus the exogenous business context (promo flags, robotics throughput) and
gets back a structured diagnosis — `cause`, `confidence`, `rationale`,
`recommended_action` — using `claude-opus-4-8` with adaptive thinking and a JSON schema.
When the SDK or an API key is unavailable it degrades to a rule-based explainer that
mirrors the same reasoning, so the loop always produces an actionable answer.

### 2. Model "confidence decay" score (`drift/confidence.py`)
A single number in `[0, 1]` (0 = as trustworthy as at deployment, 1 = collapsed) that
blends **rolling forecast-error growth** with **prediction-interval coverage breach**.
This is what lets the platform retrain when performance is genuinely eroding, even if the
raw input distribution alone hasn't tripped a statistical alert.

### 3. Auto-SLO renegotiation for inference latency (`healing/slo.py`)
A fixed latency SLO either pages constantly or hides regressions. `LatencySLO.renegotiate()`
tunes the p95 target from observed percentiles within guard rails — tightening when the
service is comfortably fast, relaxing (up to a hard ceiling) when a sustained slowdown
makes the current target unrealistic, e.g. the robotics-slowdown scenario.

---

## Architecture

```
src/self_healing_ml/
├── config.py                 thresholds (PSI/KS, confidence, SLO, validation gate)
├── data/generator.py         synthetic demand + injectable drift events
├── models/forecaster.py      ridge forecaster (fit / predict / predict_interval), stdlib
├── drift/
│   ├── detector.py           PSI + KS-style statistical drift  → DriftReport
│   ├── confidence.py         confidence-decay score            → ConfidenceReport
│   └── semantic.py           LLM cause diagnosis (+ fallback)  → SemanticDiagnosis
├── registry/model_registry.py  MLflow-style versions/stages/lineage/rollback
├── observability/metrics.py    Prometheus-style counters/gauges/histograms + exposition
├── healing/
│   ├── slo.py                auto-SLO renegotiation
│   └── orchestrator.py       the self-healing state machine (ties it all together)
├── scenario.py               the end-to-end demo scenario
└── cli.py                    `python -m self_healing_ml.cli ...`
```

### Stack integration
Real Kubeflow/Airflow/Step Functions/Prometheus/Datadog clusters can't run here, so the
platform ships the manifests that wire the **same core loop** into that ecosystem —
each delegates back into `self_healing_ml`, so there's one source of truth:

- `pipelines/airflow_dag.py` — daily batch-retraining DAG (one run == one cycle)
- `pipelines/kubeflow_pipeline.py` — multi-model KFP pipeline (fan-out across SKUs)
- `pipelines/step_functions.json` — retrain → validate → deploy → canary → **rollback** ASL
- `observability/prometheus_rules.yml` — drift / latency / rollback alerting
- `observability/grafana_dashboard.json` — drift, confidence, latency-SLO, healing panels

See [`pipelines/README.md`](pipelines/README.md) for the loop → manifest mapping.

| Stack piece | Where it maps |
|---|---|
| **MLflow** — lineage + versioning | `registry/model_registry.py` |
| **Kubeflow** — multi-model pipelines | `pipelines/kubeflow_pipeline.py` |
| **Airflow** — batch retraining schedules | `pipelines/airflow_dag.py` |
| **Step Functions** — retrain→validate→deploy | `pipelines/step_functions.json` |
| **Prometheus/Grafana** — drift + latency metrics | `observability/` |
| **Datadog** — infra-cost anomalies | robotics-slowdown → SLO/infra remediation path |

---

## Results (from the reference scenario)

Running `python examples/run_demo.py` on the built-in scenario:

- **Forecast accuracy ↑** — the seasonality-shift retrain produces a challenger that
  beats the incumbent by **~50% MAE** on the drifted window before it is promoted (the
  validation gate blocks any challenger that doesn't clear a minimum improvement).
- **Rollback time → one call** — `orchestrator.rollback()` restores the last
  known-good production model instantly (`shml_rollbacks_total`).
- **Human intervention ↓ to zero** — all four cycles diagnose, decide, and act
  automatically; the full episode is captured in an audit trail (`orchestrator.audit`).

> These numbers come from the deterministic synthetic scenario and are meant to
> illustrate the mechanism, not to benchmark a production model.

---

## Using the platform in code

```python
from self_healing_ml import (
    SelfHealingOrchestrator, generate_demand, DriftEvent,
)

series = generate_demand(
    n_days=210,
    events=[DriftEvent("seasonality_shift", 120, 150)],
    trend=0.02, yearly_amplitude=8.0, seed=7,
)

orch = SelfHealingOrchestrator()          # use_llm=None → LLM if key present, else fallback
orch.bootstrap(series, train_end=90)       # train + promote the initial model

report = orch.run_cycle(series, 120, 150, latencies_ms=[80.0] * 100)
print(report.summary())                    # drift, confidence, diagnosed cause, action taken
print(orch.metrics.render())               # Prometheus exposition
```

---

## Testing

52 tests cover the generator, forecaster, drift/confidence/SLO math, registry
lineage/rollback, the Prometheus metrics, the semantic fallback, and the full
orchestrator scenario.

```bash
python tests/run_tests.py     # zero-dependency runner (installs a tiny pytest shim if needed)
pytest tests/                 # if pytest is installed (the tests are pytest-compatible)
```

## License

MIT — see [LICENSE](LICENSE).
