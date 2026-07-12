"""A lightweight, dependency-free demand forecaster.

This is a ridge-regularized linear model over lag and calendar features, solved
with a small pure-Python Gaussian-elimination routine. It is deliberately simple
and fast to (re)train — the point of the platform is the *self-healing loop*
around the model, not the model itself. A production deployment would swap this
class for an XGBoost / Prophet / temporal-fusion model behind the same
``fit`` / ``predict`` / ``predict_interval`` interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


def _solve(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve ``matrix @ x = rhs`` via Gaussian elimination with partial pivoting."""
    n = len(matrix)
    # Build augmented matrix (copy so we don't mutate the caller's data).
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[col][col] += 1e-9  # nudge a singular pivot; ridge term usually prevents this
            pivot = col
        a[col], a[pivot] = a[pivot], a[col]
        piv = a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / piv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def _features(lag1: float, lag7: float, dow: int, promo: int, throughput: float) -> List[float]:
    """Feature vector for one observation (bias handled separately)."""
    # cyclical encoding of day-of-week keeps the feature space small and smooth
    return [
        1.0,  # bias
        lag1,
        lag7,
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        float(promo),
        float(throughput),
    ]


@dataclass
class DemandForecaster:
    """Ridge-regularized linear forecaster with residual-based prediction intervals."""

    l2: float = 1.0
    coef_: List[float] = field(default_factory=list)
    residual_std_: float = 0.0
    n_features_: int = 7
    trained_on_: int = 0

    def fit(
        self,
        values: Sequence[float],
        day_of_week: Sequence[int],
        promo_flag: Sequence[int],
        throughput: Sequence[float],
    ) -> "DemandForecaster":
        rows: List[List[float]] = []
        targets: List[float] = []
        for i in range(7, len(values)):
            rows.append(
                _features(values[i - 1], values[i - 7], day_of_week[i], promo_flag[i], throughput[i])
            )
            targets.append(values[i])
        if not rows:
            raise ValueError("need at least 8 observations to fit the forecaster")

        p = len(rows[0])
        # Normal equations with ridge: (XᵀX + λI) β = Xᵀy  (bias term unregularized)
        xtx = [[0.0] * p for _ in range(p)]
        xty = [0.0] * p
        for row, y in zip(rows, targets):
            for a in range(p):
                xty[a] += row[a] * y
                for b in range(p):
                    xtx[a][b] += row[a] * row[b]
        for a in range(1, p):  # skip bias (index 0)
            xtx[a][a] += self.l2

        self.coef_ = _solve(xtx, xty)
        self.n_features_ = p
        self.trained_on_ = len(rows)

        # residual std for prediction intervals
        sq = 0.0
        for row, y in zip(rows, targets):
            pred = sum(c * f for c, f in zip(self.coef_, row))
            sq += (y - pred) ** 2
        self.residual_std_ = math.sqrt(sq / max(1, len(rows)))
        return self

    def _predict_one(
        self, lag1: float, lag7: float, dow: int, promo: int, throughput: float
    ) -> float:
        if not self.coef_:
            raise RuntimeError("model is not fitted")
        feat = _features(lag1, lag7, dow, promo, throughput)
        return sum(c * f for c, f in zip(self.coef_, feat))

    def predict(
        self,
        values: Sequence[float],
        day_of_week: Sequence[int],
        promo_flag: Sequence[int],
        throughput: Sequence[float],
    ) -> List[float]:
        """One-step-ahead predictions aligned to indices ``[7, len(values))``."""
        out: List[float] = []
        for i in range(7, len(values)):
            out.append(
                self._predict_one(
                    values[i - 1], values[i - 7], day_of_week[i], promo_flag[i], throughput[i]
                )
            )
        return out

    def predict_interval(
        self,
        values: Sequence[float],
        day_of_week: Sequence[int],
        promo_flag: Sequence[int],
        throughput: Sequence[float],
        z: float = 1.645,  # ~90% for a normal residual
    ) -> List[Tuple[float, float, float]]:
        """Return ``(point, lower, upper)`` triples for a prediction interval."""
        margin = z * self.residual_std_
        return [(p, p - margin, p + margin) for p in self.predict(values, day_of_week, promo_flag, throughput)]


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error, guarding against zero actuals."""
    pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    if not pairs:
        return 0.0
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs)
