"""LLM-based *semantic* drift explanation — the platform's novel twist.

Statistical detectors (``detector.py``) tell you a distribution moved. They do
not tell you **why**, and "why" is what decides the right remediation:

    - a **promotion** spike is expected and self-resolving — you may just widen
      intervals and *not* retrain on the spike;
    - a **seasonality shift** means the model's calendar features are stale —
      retrain on recent data;
    - a **robotics slowdown** is a supply-side capacity ceiling, not a demand
      change at all — the fix is an infra/SLO action, not a model retrain.

This module asks Claude to read the quantitative drift signals plus the
exogenous context and produce a structured diagnosis. When the ``anthropic`` SDK
or an API key is unavailable, it falls back to a deterministic rule-based
explainer so the platform always yields an actionable explanation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import LLM_MODEL

# Canonical causes the platform reasons about.
CAUSES = ("promotion", "seasonality_shift", "robotics_slowdown", "unknown")

# Recommended remediation per cause — consumed by the orchestrator.
REMEDIATION = {
    "promotion": "widen_intervals_no_retrain",
    "seasonality_shift": "retrain_on_recent",
    "robotics_slowdown": "infra_action_renegotiate_slo",
    "unknown": "retrain_on_recent",
}


@dataclass
class SemanticDiagnosis:
    cause: str
    confidence: float  # LLM/heuristic confidence in the cause, 0..1
    rationale: str
    recommended_action: str
    source: str  # "llm" | "rule_based"
    signals: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cause": self.cause,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "recommended_action": self.recommended_action,
            "source": self.source,
        }


_SCHEMA = {
    "type": "object",
    "properties": {
        "cause": {"type": "string", "enum": list(CAUSES)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["cause", "confidence", "rationale"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are the diagnosis agent inside a self-healing demand-forecasting "
    "platform. You are given quantitative drift signals and exogenous business "
    "context for a retail SKU. Decide the single most likely root cause of the "
    "drift and explain it in one or two sentences a data/ops engineer would act "
    "on. Choose 'promotion' for a demand spike tied to a sale, "
    "'seasonality_shift' when the weekly/seasonal pattern itself has changed, "
    "'robotics_slowdown' when fulfilment throughput has dropped and is capping "
    "realized demand, and 'unknown' only when the signals are genuinely "
    "ambiguous."
)


def _build_prompt(signals: Dict[str, object]) -> str:
    return (
        "Drift and context signals (JSON):\n"
        + json.dumps(signals, indent=2, sort_keys=True)
        + "\n\nReturn the most likely root cause, your confidence (0-1), and a "
        "short rationale grounded in the specific numbers above."
    )


def _rule_based(signals: Dict[str, object]) -> SemanticDiagnosis:
    """Deterministic fallback used when the LLM is unavailable.

    Mirrors the reasoning we want the LLM to perform, so behaviour is coherent
    with or without a key.
    """
    promo_rate = float(signals.get("live_promo_rate", 0.0))
    throughput = float(signals.get("live_min_throughput", 1.0))
    mean_shift = float(signals.get("mean_shift_pct", 0.0))
    ks = float(signals.get("ks", 0.0))

    if throughput < 0.85 and mean_shift < 0:
        cause, conf = "robotics_slowdown", 0.8
        rationale = (
            f"Robotics throughput fell to {throughput:.2f} while demand dropped "
            f"{mean_shift:.1f}%, indicating a fulfilment-capacity ceiling clipping "
            "realized demand rather than a true demand change."
        )
    elif promo_rate > 0.3 and mean_shift > 0:
        cause, conf = "promotion", 0.82
        rationale = (
            f"{promo_rate:.0%} of recent days carried a promotion flag and demand "
            f"rose {mean_shift:.1f}%, consistent with a promotional spike."
        )
    elif ks >= 0.2 and abs(mean_shift) < 15:
        cause, conf = "seasonality_shift", 0.7
        rationale = (
            f"Distributional shape moved (KS={ks:.2f}) with only a {mean_shift:.1f}% "
            "mean change, consistent with a shift in the weekly/seasonal pattern."
        )
    else:
        cause, conf = "unknown", 0.5
        rationale = (
            "Signals are ambiguous; defaulting to a conservative retrain on recent data."
        )

    return SemanticDiagnosis(
        cause=cause,
        confidence=conf,
        rationale=rationale,
        recommended_action=REMEDIATION[cause],
        source="rule_based",
        signals=signals,
    )


def _llm_diagnose(signals: Dict[str, object], model: str) -> SemanticDiagnosis:
    """Call Claude to diagnose the drift. Raises on any failure so the caller
    can fall back to the rule-based path."""
    import anthropic  # imported lazily so the package works without the SDK

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / profile
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": _build_prompt(signals)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)
    cause = data.get("cause", "unknown")
    if cause not in CAUSES:
        cause = "unknown"
    return SemanticDiagnosis(
        cause=cause,
        confidence=float(data.get("confidence", 0.5)),
        rationale=str(data.get("rationale", "")).strip(),
        recommended_action=REMEDIATION[cause],
        source="llm",
        signals=signals,
    )


def explain_drift(
    signals: Dict[str, object],
    model: str = LLM_MODEL,
    use_llm: Optional[bool] = None,
) -> SemanticDiagnosis:
    """Explain *why* drift happened.

    ``use_llm`` forces the path: ``True`` requires the SDK+key (errors bubble up
    only if the call itself fails after a successful setup), ``False`` forces the
    rule-based fallback, ``None`` (default) auto-selects — LLM when an
    ``ANTHROPIC_API_KEY`` is present and the SDK imports, else rule-based.
    """
    if use_llm is False:
        return _rule_based(signals)

    key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_llm is None and not key_present:
        return _rule_based(signals)

    try:
        return _llm_diagnose(signals, model)
    except Exception:
        # Any failure (missing SDK, network, parse) degrades gracefully.
        return _rule_based(signals)


def build_signals(
    drift_report,
    confidence_report,
    live_promo_rate: float,
    live_min_throughput: float,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Assemble the compact signal dict passed to the explainer.

    Accepts the ``DriftReport`` / ``ConfidenceReport`` dataclasses (duck-typed)
    so callers don't need to marshal them by hand.
    """
    signals: Dict[str, object] = {
        "psi": getattr(drift_report, "psi", None),
        "ks": getattr(drift_report, "ks", None),
        "severity": getattr(drift_report, "severity", None),
        "mean_shift_pct": getattr(drift_report, "mean_shift_pct", None),
        "confidence_score": getattr(confidence_report, "score", None),
        "coverage": getattr(confidence_report, "coverage", None),
        "error_growth": getattr(confidence_report, "error_growth", None),
        "live_promo_rate": round(live_promo_rate, 3),
        "live_min_throughput": round(live_min_throughput, 3),
    }
    if extra:
        signals.update(extra)
    return signals
