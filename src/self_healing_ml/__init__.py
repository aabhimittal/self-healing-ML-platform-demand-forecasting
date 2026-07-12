"""Self-Healing ML Platform for Demand Forecasting.

A dependency-free reference implementation of a forecasting platform that
diagnoses its own degradation, explains *why* drift happened using an LLM agent,
retrains and redeploys behind a validation gate, and renegotiates its inference
latency SLO — with rollback in a single call.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import PlatformConfig, DEFAULT_CONFIG, LLM_MODEL
from .data.generator import DemandSeries, DriftEvent, generate_demand
from .models.forecaster import DemandForecaster, mean_absolute_error, mape
from .drift.detector import detect_drift, DriftReport
from .drift.confidence import confidence_decay, ConfidenceReport
from .drift.semantic import explain_drift, build_signals, SemanticDiagnosis
from .registry.model_registry import ModelRegistry, ModelVersion
from .observability.metrics import MetricsRegistry
from .healing.slo import LatencySLO, SLODecision
from .healing.orchestrator import SelfHealingOrchestrator, CycleReport

__all__ = [
    "__version__",
    "PlatformConfig",
    "DEFAULT_CONFIG",
    "LLM_MODEL",
    "DemandSeries",
    "DriftEvent",
    "generate_demand",
    "DemandForecaster",
    "mean_absolute_error",
    "mape",
    "detect_drift",
    "DriftReport",
    "confidence_decay",
    "ConfidenceReport",
    "explain_drift",
    "build_signals",
    "SemanticDiagnosis",
    "ModelRegistry",
    "ModelVersion",
    "MetricsRegistry",
    "LatencySLO",
    "SLODecision",
    "SelfHealingOrchestrator",
    "CycleReport",
]
