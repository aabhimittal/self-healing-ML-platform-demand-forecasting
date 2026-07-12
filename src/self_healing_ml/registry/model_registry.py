"""An MLflow-style in-memory model registry with lineage and rollback.

Tracks model versions, their metrics, the stage each occupies
(``none`` / ``staging`` / ``production`` / ``archived``), and a full transition
history. This is what makes "rollback in minutes" a single call: promoting a new
version archives the incumbent, and ``rollback`` restores the last known-good
production version instantly.

A real deployment would back this with the MLflow tracking + model registry
APIs; the interface here mirrors the concepts (versions, stages, transitions,
lineage) so the swap is mechanical.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


STAGES = ("none", "staging", "production", "archived")


@dataclass
class ModelVersion:
    version: int
    model: Any  # the fitted estimator (opaque to the registry)
    metrics: Dict[str, float] = field(default_factory=dict)
    stage: str = "none"
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent_version: Optional[int] = None  # lineage pointer


@dataclass
class Transition:
    version: int
    from_stage: str
    to_stage: str
    reason: str
    at: float = field(default_factory=time.time)


class ModelRegistry:
    def __init__(self, name: str = "demand_forecaster") -> None:
        self.name = name
        self._versions: Dict[int, ModelVersion] = {}
        self._counter = 0
        self.history: List[Transition] = []

    # --- registration -------------------------------------------------------
    def register(
        self,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        parent_version: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        self._counter += 1
        mv = ModelVersion(
            version=self._counter,
            model=model,
            metrics=dict(metrics or {}),
            parent_version=parent_version,
            tags=dict(tags or {}),
        )
        self._versions[mv.version] = mv
        return mv

    # --- stage transitions --------------------------------------------------
    def _transition(self, version: int, to_stage: str, reason: str) -> None:
        if to_stage not in STAGES:
            raise ValueError(f"unknown stage {to_stage!r}")
        mv = self._versions[version]
        self.history.append(Transition(version, mv.stage, to_stage, reason))
        mv.stage = to_stage

    def promote(self, version: int, reason: str = "validated") -> ModelVersion:
        """Promote a version to production, archiving the current production one."""
        current = self.production()
        if current and current.version != version:
            self._transition(current.version, "archived", f"superseded_by_v{version}")
        self._transition(version, "production", reason)
        return self._versions[version]

    def stage(self, version: int, reason: str = "challenger") -> ModelVersion:
        self._transition(version, "staging", reason)
        return self._versions[version]

    def rollback(self, reason: str = "auto_rollback") -> Optional[ModelVersion]:
        """Restore the most recent archived production model as production.

        Returns the restored version, or ``None`` if there is nothing to roll
        back to. The currently-broken production model is archived.
        """
        current = self.production()
        # find the most recently archived version that had reached production
        archived = [
            t for t in reversed(self.history)
            if t.to_stage == "archived" and t.from_stage == "production"
        ]
        if not archived:
            return None
        target = archived[0].version
        if current:
            self._transition(current.version, "archived", reason + "_demote_current")
        self._transition(target, "production", reason)
        return self._versions[target]

    # --- queries ------------------------------------------------------------
    def production(self) -> Optional[ModelVersion]:
        for mv in self._versions.values():
            if mv.stage == "production":
                return mv
        return None

    def get(self, version: int) -> ModelVersion:
        return self._versions[version]

    def versions(self) -> List[ModelVersion]:
        return list(self._versions.values())

    def lineage(self, version: int) -> List[int]:
        """Return the ancestry chain [root, ..., version]."""
        chain: List[int] = []
        v: Optional[int] = version
        seen = set()
        while v is not None and v not in seen:
            seen.add(v)
            chain.append(v)
            v = self._versions[v].parent_version
        return list(reversed(chain))
