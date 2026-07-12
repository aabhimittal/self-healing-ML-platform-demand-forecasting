"""Central configuration for the self-healing ML platform.

All thresholds live here so the self-healing loop, the drift detectors, and the
observability layer share a single source of truth. Values are intentionally
conservative defaults suitable for a demand-forecasting workload with daily
batch retraining.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# The model used by the LLM semantic-drift explainer. Kept here so the whole
# platform references one identifier.
LLM_MODEL = "claude-opus-4-8"


@dataclass(frozen=True)
class DriftThresholds:
    """Thresholds that decide when a statistical drift signal is actionable.

    PSI (Population Stability Index) buckets follow the industry convention:
      < 0.1  no significant shift
      0.1-0.25 moderate shift (watch)
      > 0.25 major shift (act)
    """

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_alert: float = 0.20  # KS-style max CDF distance
    bins: int = 10


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Thresholds for the model 'confidence decay' score (0=healthy, 1=collapsed)."""

    warn: float = 0.35
    heal: float = 0.55
    # Window (in observations) used to compute rolling error growth.
    window: int = 14
    # Nominal prediction-interval coverage we expect a healthy model to hold.
    target_coverage: float = 0.90


@dataclass(frozen=True)
class SLOConfig:
    """Auto-SLO renegotiation parameters for inference latency (milliseconds)."""

    initial_p95_ms: float = 120.0
    # Renegotiate only when observed latency drifts beyond this fraction of the SLO.
    renegotiation_band: float = 0.15
    # Never relax the SLO past this hard ceiling (protects downstream consumers).
    hard_ceiling_ms: float = 400.0
    # Never tighten below this floor (avoids flapping on noise).
    floor_ms: float = 40.0
    # Percentiles tracked for renegotiation decisions.
    percentiles: tuple = (0.95, 0.99)


@dataclass(frozen=True)
class ValidationGate:
    """Champion/challenger gate used before a retrained model is promoted."""

    # A challenger must beat the incumbent MAE by at least this fraction to deploy.
    min_relative_improvement: float = 0.02
    # Absolute MAE ceiling; a challenger above this can never be promoted.
    max_acceptable_mae: float = 1e9


@dataclass(frozen=True)
class PlatformConfig:
    drift: DriftThresholds = field(default_factory=DriftThresholds)
    confidence: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    slo: SLOConfig = field(default_factory=SLOConfig)
    validation: ValidationGate = field(default_factory=ValidationGate)
    llm_model: str = LLM_MODEL
    # Reference/live window sizes (observations) for drift comparison.
    reference_window: int = 60
    live_window: int = 30

    def as_dict(self) -> Dict[str, object]:
        return {
            "llm_model": self.llm_model,
            "reference_window": self.reference_window,
            "live_window": self.live_window,
            "psi_alert": self.drift.psi_alert,
            "confidence_heal": self.confidence.heal,
            "slo_initial_p95_ms": self.slo.initial_p95_ms,
        }


DEFAULT_CONFIG = PlatformConfig()
