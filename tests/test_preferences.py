"""Tests for preference learning module."""
import numpy as np
import pytest

from ads_agent.preferences.preference_model import (
    PreferencePair,
    LearnedReward,
    BradleyTerryModel,
    generate_synthetic_preferences,
)


# ============================================================================
# Bradley-Terry Model
# ============================================================================

class TestBradleyTerryModel:
    def test_fit_empty_pairs(self):
        model = BradleyTerryModel(feature_dim=3)
        result = model.fit([])
        assert result.n_pairs == 0
        np.testing.assert_array_equal(result.weights, np.zeros(3))

    def test_fit_synthetic_data(self):
        pairs, true_weights = generate_synthetic_preferences(
            n_pairs=200, feature_dim=4, seed=42,
        )
        model = BradleyTerryModel(feature_dim=4)
        result = model.fit(pairs, max_iter=200)

        assert result.n_pairs == 200
        assert result.accuracy > 0.7  # should learn reasonably well

    def test_weight_direction_recovery(self):
        """Learned weights should correlate with true weights."""
        true_w = np.array([1.0, -0.5, 0.3, 0.0])
        pairs, _ = generate_synthetic_preferences(
            n_pairs=500, feature_dim=4,
            true_weights=true_w, noise=0.05, seed=42,
        )
        model = BradleyTerryModel(feature_dim=4, regularization=0.001)
        result = model.fit(pairs, max_iter=300, lr=0.05)

        # Cosine similarity between learned and true weights
        cos_sim = np.dot(result.weights, true_w) / (
            np.linalg.norm(result.weights) * np.linalg.norm(true_w)
        )
        assert cos_sim > 0.8  # should be correlated

    def test_predict_preference(self):
        pairs, _ = generate_synthetic_preferences(n_pairs=200, feature_dim=3, seed=42)
        model = BradleyTerryModel(feature_dim=3)
        model.fit(pairs)

        f_a = np.array([1, 0, 0])
        f_b = np.array([-1, 0, 0])
        prob = model.predict_preference(f_a, f_b)
        assert 0.0 <= prob <= 1.0

    def test_predict_before_fit_raises(self):
        model = BradleyTerryModel(feature_dim=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_preference(np.zeros(3), np.zeros(3))


# ============================================================================
# LearnedReward
# ============================================================================

class TestLearnedReward:
    def test_score(self):
        reward = LearnedReward(
            weights=np.array([1.0, 2.0, 3.0]),
            feature_names=["a", "b", "c"],
            n_pairs=10, log_likelihood=-5.0, accuracy=0.8,
        )
        assert reward.score(np.array([1, 0, 0])) == 1.0
        assert reward.score(np.array([0, 0, 1])) == 3.0

    def test_rank(self):
        reward = LearnedReward(
            weights=np.array([1.0, 0.0]),
            feature_names=["x", "y"],
            n_pairs=10, log_likelihood=-5.0, accuracy=0.8,
        )
        items = [np.array([0.5, 0]), np.array([2.0, 0]), np.array([1.0, 0])]
        ranking = reward.rank(items)
        assert ranking[0] == 1  # highest score (2.0)
        assert ranking[-1] == 0  # lowest score (0.5)


# ============================================================================
# Synthetic Preferences
# ============================================================================

class TestSyntheticPreferences:
    def test_generate_correct_count(self):
        pairs, weights = generate_synthetic_preferences(n_pairs=50)
        assert len(pairs) == 50
        assert len(weights) > 0

    def test_generate_consistent_preference_direction(self):
        """In each pair, item_a should have higher reward than item_b."""
        true_w = np.array([1.0, 0.5])
        pairs, _ = generate_synthetic_preferences(
            n_pairs=100, feature_dim=2,
            true_weights=true_w, noise=0.0, seed=42,
        )
        for p in pairs:
            r_a = true_w @ p.features_a
            r_b = true_w @ p.features_b
            assert r_a >= r_b  # no noise → deterministic preference
