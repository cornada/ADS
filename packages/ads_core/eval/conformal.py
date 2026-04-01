"""Conformal prediction for curriculum recommendations.

Provides guaranteed prediction sets: "with probability 1-α, the true
outcome is in this set." Unlike point predictions, conformal prediction
gives honest uncertainty quantification with finite-sample guarantees.

Educational applications:
- Course difficulty prediction: "with 90% probability, this course's
  difficulty for student X is in [easy, medium]"
- Market fit prediction: "with 90% probability, this curriculum's
  market alignment score is in [0.6, 0.8]"
- Competency coverage: "with 90% probability, completing this track
  covers competencies {A, B, C}"

The key advantage: NO distributional assumptions. Works with any
base predictor (neural net, KNN, random forest, etc.).

Method:
1. Split data into train + calibration sets
2. Train base predictor on train set
3. Compute nonconformity scores on calibration set
4. For new point: prediction set = {y : score(x, y) ≤ quantile}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class PredictionSet:
    """A conformal prediction set for a single instance."""

    point_prediction: float          # base predictor output
    lower: float                     # lower bound of prediction interval
    upper: float                     # upper bound of prediction interval
    alpha: float                     # miscoverage rate
    coverage_guarantee: float        # 1 - alpha
    set_size: float                  # width of prediction interval
    metadata: Dict[str, Any] = field(default_factory=dict)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass
class ClassPredictionSet:
    """A conformal prediction set for classification."""

    predicted_class: str
    prediction_set: List[str]        # classes in the prediction set
    scores: Dict[str, float]         # nonconformity score per class
    alpha: float
    set_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def contains(self, label: str) -> bool:
        return label in self.prediction_set


@dataclass
class ConformalResult:
    """Aggregated conformal prediction results."""

    n_test: int
    alpha: float
    empirical_coverage: float       # should be ≥ 1 - alpha
    mean_set_size: float            # average prediction set size
    median_set_size: float
    coverage_by_class: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if empirical coverage meets guarantee."""
        return self.empirical_coverage >= 1 - self.alpha - 0.05  # small tolerance


# ============================================================================
# Split Conformal Prediction (Regression)
# ============================================================================

class SplitConformalRegressor:
    """Split conformal prediction for regression.

    Uses absolute residual as nonconformity score:
    score(x, y) = |y - f(x)|

    The prediction interval for new x is:
    [f(x) - q, f(x) + q]
    where q = (1-α)(1+1/n)-quantile of calibration scores.
    """

    def __init__(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.1,
    ) -> None:
        self.predictor = predictor
        self.alpha = alpha
        self._calibration_scores: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Calibrate using held-out calibration data."""
        predictions = self.predictor(X_cal)
        self._calibration_scores = np.abs(y_cal - predictions)

        n = len(self._calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(self._calibration_scores, q_level))

    def predict(self, X: np.ndarray) -> List[PredictionSet]:
        """Produce prediction sets for new data."""
        if self._quantile is None:
            raise RuntimeError("Must call calibrate() first")

        predictions = self.predictor(X)
        results = []
        for i in range(len(X)):
            pred = float(predictions[i])
            results.append(PredictionSet(
                point_prediction=pred,
                lower=pred - self._quantile,
                upper=pred + self._quantile,
                alpha=self.alpha,
                coverage_guarantee=1 - self.alpha,
                set_size=2 * self._quantile,
            ))
        return results

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ConformalResult:
        """Evaluate conformal coverage on test data."""
        pred_sets = self.predict(X_test)
        covered = sum(1 for ps, y in zip(pred_sets, y_test) if ps.contains(y))
        sizes = [ps.set_size for ps in pred_sets]

        return ConformalResult(
            n_test=len(y_test),
            alpha=self.alpha,
            empirical_coverage=covered / len(y_test),
            mean_set_size=float(np.mean(sizes)),
            median_set_size=float(np.median(sizes)),
        )


# ============================================================================
# Conformal Classification (LAC — Least Ambiguous Classifier)
# ============================================================================

class ConformalClassifier:
    """Conformal prediction for classification using softmax scores.

    Uses the LAC (Least Ambiguous set-valued Classifier) approach:
    score(x, y) = 1 - P(Y=y|X=x)

    Prediction set includes all classes y where score ≤ quantile.
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        classes: List[str],
        alpha: float = 0.1,
    ) -> None:
        """
        Args:
            score_fn: Function returning (n, C) probability matrix
            classes: List of class names
            alpha: Miscoverage rate
        """
        self.score_fn = score_fn
        self.classes = classes
        self.alpha = alpha
        self._quantile: Optional[float] = None

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Calibrate using held-out data."""
        probs = self.score_fn(X_cal)  # (n, C)
        n = len(y_cal)

        # Nonconformity score: 1 - P(true class)
        scores = np.zeros(n)
        for i in range(n):
            class_idx = self.classes.index(y_cal[i])
            scores[i] = 1 - probs[i, class_idx]

        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(scores, q_level))

    def predict(self, X: np.ndarray) -> List[ClassPredictionSet]:
        """Produce class prediction sets."""
        if self._quantile is None:
            raise RuntimeError("Must call calibrate() first")

        probs = self.score_fn(X)
        results = []

        for i in range(len(X)):
            scores = {
                cls: 1 - probs[i, j]
                for j, cls in enumerate(self.classes)
            }
            pred_set = [cls for cls, s in scores.items() if s <= self._quantile]
            predicted = self.classes[int(np.argmax(probs[i]))]

            results.append(ClassPredictionSet(
                predicted_class=predicted,
                prediction_set=pred_set,
                scores=scores,
                alpha=self.alpha,
                set_size=len(pred_set),
            ))

        return results

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ConformalResult:
        """Evaluate conformal coverage on test data."""
        pred_sets = self.predict(X_test)
        covered = sum(1 for ps, y in zip(pred_sets, y_test) if ps.contains(y))
        sizes = [ps.set_size for ps in pred_sets]

        # Per-class coverage
        class_counts: Dict[str, List[bool]] = {c: [] for c in self.classes}
        for ps, y in zip(pred_sets, y_test):
            if y in class_counts:
                class_counts[y].append(ps.contains(y))

        coverage_by_class = {
            c: float(np.mean(hits)) if hits else 0.0
            for c, hits in class_counts.items()
        }

        return ConformalResult(
            n_test=len(y_test),
            alpha=self.alpha,
            empirical_coverage=covered / len(y_test),
            mean_set_size=float(np.mean(sizes)),
            median_set_size=float(np.median(sizes)),
            coverage_by_class=coverage_by_class,
        )


# ============================================================================
# Adaptive Conformal (locally weighted)
# ============================================================================

class AdaptiveConformalRegressor:
    """Locally adaptive conformal prediction.

    Uses normalized residuals: score(x, y) = |y - f(x)| / σ(x)
    where σ(x) is a local uncertainty estimate. This produces
    prediction intervals that are tighter where the model is confident
    and wider where it is uncertain.
    """

    def __init__(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        uncertainty_fn: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.1,
    ) -> None:
        self.predictor = predictor
        self.uncertainty_fn = uncertainty_fn
        self.alpha = alpha
        self._quantile: Optional[float] = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        predictions = self.predictor(X_cal)
        sigmas = self.uncertainty_fn(X_cal)
        sigmas = np.maximum(sigmas, 1e-8)

        scores = np.abs(y_cal - predictions) / sigmas

        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(scores, q_level))

    def predict(self, X: np.ndarray) -> List[PredictionSet]:
        if self._quantile is None:
            raise RuntimeError("Must call calibrate() first")

        predictions = self.predictor(X)
        sigmas = self.uncertainty_fn(X)
        sigmas = np.maximum(sigmas, 1e-8)

        results = []
        for i in range(len(X)):
            pred = float(predictions[i])
            width = self._quantile * sigmas[i]
            results.append(PredictionSet(
                point_prediction=pred,
                lower=pred - width,
                upper=pred + width,
                alpha=self.alpha,
                coverage_guarantee=1 - self.alpha,
                set_size=2 * width,
                metadata={"local_sigma": float(sigmas[i])},
            ))
        return results
