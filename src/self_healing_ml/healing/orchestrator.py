"""The self-healing orchestrator — the platform's control loop.

Ties the pieces together into one state machine:

    detect (statistical drift + confidence decay)
      -> diagnose (LLM semantic explanation of *why*)
        -> decide (remediation from the diagnosed cause)
          -> act (retrain + champion/challenger validate + promote, OR widen
                  intervals, OR renegotiate the latency SLO)
            -> rollback in one call if a promoted model regresses

Every step emits Prometheus-style metrics and appends to an audit trail, so the
whole episode is inspectable — this is what replaces a human watching dashboards.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import PlatformConfig, DEFAULT_CONFIG
from ..data.generator import DemandSeries
from ..models.forecaster import DemandForecaster, mean_absolute_error
from ..drift.detector import detect_drift, DriftReport
from ..drift.confidence import confidence_decay, ConfidenceReport
from ..drift.semantic import explain_drift, build_signals, SemanticDiagnosis
from ..registry.model_registry import ModelRegistry, ModelVersion
from ..observability.metrics import MetricsRegistry
from .slo import LatencySLO, SLODecision


@dataclass
class CycleReport:
    cycle: int
    drift: DriftReport
    confidence: ConfidenceReport
    slo: SLODecision
    diagnosis: Optional[SemanticDiagnosis]
    action: str
    healed: bool
    promoted_version: Optional[int]
    production_version: int
    notes: str = ""

    def summary(self) -> str:
        cause = self.diagnosis.cause if self.diagnosis else "n/a"
        return (
            f"cycle={self.cycle} drift={self.drift.severity} "
            f"confidence={self.confidence.score:.2f}({self.confidence.status}) "
            f"cause={cause} action={self.action} "
            f"prod_v={self.production_version} healed={self.healed}"
        )


@dataclass
class AuditEvent:
    at: float
    kind: str
    detail: Dict[str, object]


class SelfHealingOrchestrator:
    def __init__(
        self,
        config: PlatformConfig = DEFAULT_CONFIG,
        registry: Optional[ModelRegistry] = None,
        metrics: Optional[MetricsRegistry] = None,
        use_llm: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.registry = registry or ModelRegistry()
        self.metrics = metrics or MetricsRegistry()
        self.slo = LatencySLO(config.slo)
        self.use_llm = use_llm
        self.audit: List[AuditEvent] = []
        self._baseline_mae: float = 0.0
        self._cycle = 0

    # --- setup --------------------------------------------------------------
    def bootstrap(self, series: DemandSeries, train_end: int) -> ModelVersion:
        """Train the initial production model on ``series[:train_end]``."""
        model = self._train(series, 0, train_end)
        actual, pred, _ = self._score_window(model, series, 7, train_end)
        self._baseline_mae = mean_absolute_error(actual, pred)
        mv = self.registry.register(
            model,
            metrics={"mae": round(self._baseline_mae, 4)},
            tags={"origin": "bootstrap"},
        )
        self.registry.promote(mv.version, reason="bootstrap")
        self.metrics.set("shml_production_version", mv.version, help="Current production model version")
        self.metrics.set("shml_baseline_mae", self._baseline_mae, help="Baseline MAE at deployment")
        self._audit("bootstrap", {"version": mv.version, "baseline_mae": self._baseline_mae})
        return mv

    # --- one self-healing cycle --------------------------------------------
    def run_cycle(
        self,
        series: DemandSeries,
        live_start: int,
        live_end: int,
        latencies_ms: Sequence[float],
    ) -> CycleReport:
        self._cycle += 1
        cfg = self.config
        prod = self.registry.production()
        assert prod is not None, "call bootstrap() before run_cycle()"

        # 1) statistical drift: reference window vs live window ---------------
        ref_start = max(0, live_start - cfg.reference_window)
        reference = series.window(ref_start, live_start)
        live = series.window(live_start, live_end)
        drift = detect_drift(reference, live, cfg.drift)
        self.metrics.set("shml_drift_psi", drift.psi, help="Population Stability Index")
        self.metrics.set("shml_drift_ks", drift.ks, help="KS distance")

        # 2) confidence decay of the production model on the live window ------
        actual, pred, intervals = self._score_window(prod.model, series, live_start, live_end)
        conf = confidence_decay(self._baseline_mae, actual, pred, intervals, cfg.confidence)
        self.metrics.set("shml_confidence_decay", conf.score, help="Model confidence decay score")

        # 3) auto-SLO renegotiation from observed latency --------------------
        slo_decision = self.slo.renegotiate(latencies_ms)
        for lat in latencies_ms:
            self.metrics.observe("shml_inference_latency_ms", lat, help="Inference latency (ms)")
        self.metrics.set("shml_slo_p95_ms", self.slo.p95_target_ms, help="Current p95 latency SLO")
        if slo_decision.changed:
            self.metrics.inc("shml_slo_renegotiations_total", help="SLO renegotiation count")
            self._audit("slo_renegotiated", {
                "action": slo_decision.action,
                "old_ms": slo_decision.old_p95_ms,
                "new_ms": slo_decision.new_p95_ms,
                "reason": slo_decision.reason,
            })

        # 4) decide whether to act -------------------------------------------
        # Only self-heal when the model's *own reliability* is affected. Pure
        # input drift while the model is still accurate (confidence healthy) is
        # logged but not acted on — retraining an accurate model is wasteful and
        # risks regressions. We act when confidence has decayed to the heal
        # threshold, or when a statistical drift alert coincides with any
        # measurable confidence erosion.
        needs_action = conf.should_heal or (drift.drifted and conf.status != "healthy")
        diagnosis: Optional[SemanticDiagnosis] = None
        action = "monitor"
        healed = False
        promoted_version: Optional[int] = None
        notes = ""

        if needs_action:
            promo_rate = sum(series.promo_flag[live_start:live_end]) / max(1, live_end - live_start)
            min_tput = min(series.robotics_throughput[live_start:live_end], default=1.0)
            signals = build_signals(drift, conf, promo_rate, min_tput)
            diagnosis = explain_drift(signals, model=cfg.llm_model, use_llm=self.use_llm)
            self.metrics.inc("shml_drift_diagnoses_total", cause=diagnosis.cause,
                             help="Semantic drift diagnoses by cause")
            self._audit("diagnosis", diagnosis.as_dict())

            action = diagnosis.recommended_action
            if action == "retrain_on_recent":
                promoted_version, healed, notes = self._retrain_and_validate(
                    series, live_start, live_end, prod, diagnosis
                )
            elif action == "widen_intervals_no_retrain":
                # promotion spike: expected, self-resolving — don't retrain on it.
                notes = "expected promotional spike; widened intervals, no retrain"
                self.metrics.inc("shml_actions_total", action="widen_intervals")
                self._audit("action_widen_intervals", {"cause": diagnosis.cause})
            elif action == "infra_action_renegotiate_slo":
                # robotics slowdown: a supply-side capacity issue, not a demand
                # change — the SLO renegotiation above is the correct remediation.
                notes = "supply-side slowdown; handled via SLO renegotiation, no retrain"
                self.metrics.inc("shml_actions_total", action="infra_slo")
                self._audit("action_infra_slo", {"cause": diagnosis.cause})

        self.metrics.set("shml_production_version", self.registry.production().version)
        report = CycleReport(
            cycle=self._cycle,
            drift=drift,
            confidence=conf,
            slo=slo_decision,
            diagnosis=diagnosis,
            action=action,
            healed=healed,
            promoted_version=promoted_version,
            production_version=self.registry.production().version,
            notes=notes,
        )
        self._audit("cycle_complete", {"summary": report.summary()})
        return report

    # --- retrain + champion/challenger validation ---------------------------
    def _retrain_and_validate(
        self,
        series: DemandSeries,
        live_start: int,
        live_end: int,
        incumbent: ModelVersion,
        diagnosis: SemanticDiagnosis,
    ) -> Tuple[Optional[int], bool, str]:
        cfg = self.config
        # Train a challenger on the most recent data (reference + live window).
        train_start = max(0, live_end - (cfg.reference_window + cfg.live_window))
        challenger = self._train(series, train_start, live_end)

        # Validate both on the live window (the segment that exhibited drift).
        inc_actual, inc_pred, _ = self._score_window(incumbent.model, series, live_start, live_end)
        ch_actual, ch_pred, _ = self._score_window(challenger, series, live_start, live_end)
        inc_mae = mean_absolute_error(inc_actual, inc_pred)
        ch_mae = mean_absolute_error(ch_actual, ch_pred)

        rel_improvement = 0.0 if inc_mae == 0 else (inc_mae - ch_mae) / inc_mae
        mv = self.registry.register(
            challenger,
            metrics={"mae": round(ch_mae, 4), "incumbent_mae": round(inc_mae, 4)},
            parent_version=incumbent.version,
            tags={"origin": "retrain", "cause": diagnosis.cause},
        )
        self.registry.stage(mv.version, reason=f"challenger_for_{diagnosis.cause}")
        self.metrics.inc("shml_retrains_total", help="Total retrains triggered")

        gate = cfg.validation
        passes = (
            rel_improvement >= gate.min_relative_improvement
            and ch_mae <= gate.max_acceptable_mae
        )
        if passes:
            self.registry.promote(mv.version, reason=f"beat_incumbent_by_{rel_improvement:.1%}")
            # rebase the baseline to the freshly validated performance
            self._baseline_mae = ch_mae
            self.metrics.set("shml_baseline_mae", self._baseline_mae)
            self.metrics.inc("shml_promotions_total", help="Total model promotions")
            self.metrics.inc("shml_actions_total", action="retrain_promote")
            self._audit("promote", {
                "version": mv.version, "challenger_mae": ch_mae,
                "incumbent_mae": inc_mae, "rel_improvement": round(rel_improvement, 4),
            })
            return mv.version, True, (
                f"retrained (cause={diagnosis.cause}); challenger v{mv.version} beat "
                f"incumbent by {rel_improvement:.1%}, promoted to production"
            )

        self.metrics.inc("shml_actions_total", action="retrain_rejected")
        self._audit("reject_challenger", {
            "version": mv.version, "challenger_mae": ch_mae,
            "incumbent_mae": inc_mae, "rel_improvement": round(rel_improvement, 4),
        })
        return None, False, (
            f"retrained (cause={diagnosis.cause}) but challenger v{mv.version} did not "
            f"clear the validation gate ({rel_improvement:.1%} < "
            f"{gate.min_relative_improvement:.1%}); kept incumbent"
        )

    # --- rollback -----------------------------------------------------------
    def rollback(self, reason: str = "manual_rollback") -> Optional[ModelVersion]:
        restored = self.registry.rollback(reason=reason)
        if restored:
            self.metrics.inc("shml_rollbacks_total", help="Total rollbacks")
            self.metrics.set("shml_production_version", restored.version)
            self._baseline_mae = restored.metrics.get("mae", self._baseline_mae)
            self.metrics.set("shml_baseline_mae", self._baseline_mae)
            self._audit("rollback", {"restored_version": restored.version, "reason": reason})
        return restored

    # --- helpers ------------------------------------------------------------
    def _train(self, series: DemandSeries, start: int, end: int) -> DemandForecaster:
        model = DemandForecaster()
        model.fit(
            series.values[start:end],
            series.day_of_week[start:end],
            series.promo_flag[start:end],
            series.robotics_throughput[start:end],
        )
        return model

    def _score_window(
        self, model: DemandForecaster, series: DemandSeries, start: int, end: int
    ) -> Tuple[List[float], List[float], List[Tuple[float, float, float]]]:
        """Predict one-step-ahead for indices ``[max(start,7), end)``.

        Uses the raw series history for the lag features so a window can be
        scored without leaking future values into the features.
        """
        begin = max(start, 7)
        actual: List[float] = []
        pred: List[float] = []
        intervals: List[Tuple[float, float, float]] = []
        for i in range(begin, end):
            triple = model.predict_interval(
                series.values[i - 7 : i + 1],
                series.day_of_week[i - 7 : i + 1],
                series.promo_flag[i - 7 : i + 1],
                series.robotics_throughput[i - 7 : i + 1],
            )
            # predict_interval over an 8-element window yields exactly one row (index 7)
            point, lo, hi = triple[-1]
            actual.append(series.values[i])
            pred.append(point)
            intervals.append((point, lo, hi))
        return actual, pred, intervals

    def _audit(self, kind: str, detail: Dict[str, object]) -> None:
        self.audit.append(AuditEvent(at=time.time(), kind=kind, detail=detail))
