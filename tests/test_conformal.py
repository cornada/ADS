"""Tests for conformal prediction module."""
import numpy as np
import pytest

from ads_core.eval.conformal import (
    PredictionSet,
    ClassPredictionSet,
    ConformalResult,
    SplitConformalRegressor,
    ConformalClassifier,
    AdaptiveConformalRegressor,
)


# ============================================================================
# Data classes
# ============================================================================

class TestPredictionSet:
    def test_contains(self):
        ps = PredictionSet(
            point_prediction=5.0,
            lower=3.0, upper=7.0,
            alpha=0.1, coverage_guarantee=0.9,
            set_size=4.0,
        )
        assert ps.contains(5.0)
        assert ps.contains(3.0)
        assert ps.contains(7.0)
        assert not ps.contains(2.9)
        assert not ps.contains(7.1)


class TestClassPredictionSet:
    def test_contains(self):
        cps = ClassPredictionSet(
            predicted_class="A",
            prediction_set=["A", "B"],
            scores={"A": 0.1, "B": 0.3, "C": 0.8},
            alpha=0.1,
            set_size=2,
        )
        assert cps.contains("A")
        assert cps.contains("B")
        assert not cps.contains("C")


class TestConformalResult:
    def test_valid_coverage(self):
        r = ConformalResult(
            n_test=100, alpha=0.1,
            empirical_coverage=0.92,
            mean_set_size=2.0, median_set_size=2.0,
        )
        assert r.is_valid()

    def test_invalid_coverage(self):
        r = ConformalResult(
            n_test=100, alpha=0.1,
            empirical_coverage=0.80,
            mean_set_size=2.0, median_set_size=2.0,
        )
        assert not r.is_valid()


# ============================================================================
# Split Conformal Regressor
# ============================================================================

class TestSplitConformalRegressor:
    @pytest.fixture
    def trained_regressor(self):
        """Create a calibrated regressor with known predictor."""
        rng = np.random.RandomState(42)
        # Predictor: f(x) = x[:, 0] (just first feature)
        predictor = lambda X: X[:, 0]

        scr = SplitConformalRegressor(predictor=predictor, alpha=0.1)

        # Calibration data with noise
        n_cal = 200
        X_cal = rng.randn(n_cal, 3)
        y_cal = X_cal[:, 0] + rng.normal(0, 0.5, n_cal)
        scr.calibrate(X_cal, y_cal)
        return scr

    def test_calibrate_sets_quantile(self, trained_regressor):
        assert trained_regressor._quantile is not None
        assert trained_regressor._quantile > 0

    def test_predict_before_calibrate_raises(self):
        scr = SplitConformalRegressor(predictor=lambda X: X[:, 0])
        with pytest.raises(RuntimeError, match="calibrate"):
            scr.predict(np.array([[1, 2, 3]]))

    def test_predict_returns_intervals(self, trained_regressor):
        X = np.array([[1.0, 0, 0], [2.0, 0, 0]])
        preds = trained_regressor.predict(X)
        assert len(preds) == 2
        for ps in preds:
            assert ps.lower < ps.upper
            assert ps.set_size > 0
            assert ps.alpha == 0.1

    def test_coverage_guarantee(self, trained_regressor):
        """Coverage should be ≥ 1-α on test data (with high probability)."""
        rng = np.random.RandomState(123)
        n_test = 500
        X_test = rng.randn(n_test, 3)
        y_test = X_test[:, 0] + rng.normal(0, 0.5, n_test)

        result = trained_regressor.evaluate(X_test, y_test)
        # Should cover at least 85% (some slack for finite-sample)
        assert result.empirical_coverage >= 0.85
        assert result.n_test == n_test

    def test_wider_alpha_means_smaller_sets(self):
        """Larger α (less coverage) should produce smaller prediction sets."""
        rng = np.random.RandomState(42)
        predictor = lambda X: X[:, 0]
        n_cal = 200
        X_cal = rng.randn(n_cal, 3)
        y_cal = X_cal[:, 0] + rng.normal(0, 0.5, n_cal)

        scr_tight = SplitConformalRegressor(predictor=predictor, alpha=0.05)
        scr_loose = SplitConformalRegressor(predictor=predictor, alpha=0.3)
        scr_tight.calibrate(X_cal, y_cal)
        scr_loose.calibrate(X_cal, y_cal)

        assert scr_loose._quantile < scr_tight._quantile


# ============================================================================
# Conformal Classifier
# ============================================================================

class TestConformalClassifier:
    @pytest.fixture
    def trained_classifier(self):
        """Create a calibrated classifier."""
        rng = np.random.RandomState(42)
        classes = ["easy", "medium", "hard"]

        # Softmax score function (simulated)
        def score_fn(X):
            raw = X @ np.array([[2, 1, 0], [0, 1, 2], [1, 1, 1]]).T
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

        cc = ConformalClassifier(score_fn=score_fn, classes=classes, alpha=0.1)

        # Calibration
        n_cal = 300
        X_cal = rng.randn(n_cal, 3)
        probs = score_fn(X_cal)
        y_cal = np.array([classes[np.argmax(probs[i])] for i in range(n_cal)])
        # Add some noise to labels
        for i in range(n_cal):
            if rng.rand() < 0.2:
                y_cal[i] = rng.choice(classes)

        cc.calibrate(X_cal, y_cal)
        return cc, score_fn, classes

    def test_calibrate_sets_quantile(self, trained_classifier):
        cc, _, _ = trained_classifier
        assert cc._quantile is not None

    def test_predict_returns_class_sets(self, trained_classifier):
        cc, _, classes = trained_classifier
        X = np.array([[1, 0, 0], [0, 1, 0]])
        preds = cc.predict(X)
        assert len(preds) == 2
        for ps in preds:
            assert ps.predicted_class in classes
            assert all(c in classes for c in ps.prediction_set)
            assert ps.set_size >= 1

    def test_coverage_guarantee(self, trained_classifier):
        cc, score_fn, classes = trained_classifier
        rng = np.random.RandomState(99)
        n_test = 500
        X_test = rng.randn(n_test, 3)
        probs = score_fn(X_test)
        y_test = np.array([classes[np.argmax(probs[i])] for i in range(n_test)])
        for i in range(n_test):
            if rng.rand() < 0.2:
                y_test[i] = rng.choice(classes)

        result = cc.evaluate(X_test, y_test)
        # Coverage should be reasonable
        assert result.empirical_coverage >= 0.80
        assert result.coverage_by_class is not None

    def test_predict_before_calibrate_raises(self):
        cc = ConformalClassifier(
            score_fn=lambda X: np.ones((len(X), 2)) * 0.5,
            classes=["A", "B"],
        )
        with pytest.raises(RuntimeError, match="calibrate"):
            cc.predict(np.array([[1, 2]]))


# ============================================================================
# Adaptive Conformal Regressor
# ============================================================================

class TestAdaptiveConformal:
    def test_adaptive_produces_variable_widths(self):
        """Adaptive intervals should vary with local uncertainty."""
        rng = np.random.RandomState(42)
        predictor = lambda X: X[:, 0]
        # Uncertainty increases with |x|
        uncertainty_fn = lambda X: np.abs(X[:, 0]) + 0.1

        acr = AdaptiveConformalRegressor(
            predictor=predictor,
            uncertainty_fn=uncertainty_fn,
            alpha=0.1,
        )

        n_cal = 200
        X_cal = rng.randn(n_cal, 2)
        y_cal = X_cal[:, 0] + rng.normal(0, 0.5, n_cal) * (np.abs(X_cal[:, 0]) + 0.1)
        acr.calibrate(X_cal, y_cal)

        # Predict for low-uncertainty and high-uncertainty points
        X_low = np.array([[0.1, 0]])  # small |x| -> small sigma
        X_high = np.array([[3.0, 0]])  # large |x| -> large sigma

        ps_low = acr.predict(X_low)[0]
        ps_high = acr.predict(X_high)[0]

        # High-uncertainty region should have wider interval
        assert ps_high.set_size > ps_low.set_size
        assert ps_low.metadata.get("local_sigma", 0) < ps_high.metadata.get("local_sigma", 0)

    def test_predict_before_calibrate_raises(self):
        acr = AdaptiveConformalRegressor(
            predictor=lambda X: X[:, 0],
            uncertainty_fn=lambda X: np.ones(len(X)),
        )
        with pytest.raises(RuntimeError, match="calibrate"):
            acr.predict(np.array([[1, 2]]))
