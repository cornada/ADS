"""Tests for IRT models and adaptive testing."""
import numpy as np
import pytest

from ads_agent.assessment.irt_model import (
    IRTItem,
    IRTResponse,
    AbilityEstimate,
    IRTModel,
    generate_item_bank,
)
from ads_agent.assessment.adaptive_selection import (
    AdaptiveSelector,
    CATSession,
    run_cat_simulation,
)


# ============================================================================
# IRTItem
# ============================================================================

class TestIRTItem:
    def test_probability_bounds(self):
        item = IRTItem(item_id="t1", difficulty=0.0, discrimination=1.0)
        for theta in np.linspace(-4, 4, 20):
            p = item.probability(theta)
            assert 0.0 <= p <= 1.0

    def test_probability_at_difficulty(self):
        """At θ = b, P should be 0.5 for 2PL."""
        item = IRTItem(item_id="t1", difficulty=1.5, discrimination=1.0)
        assert abs(item.probability(1.5) - 0.5) < 1e-10

    def test_probability_monotonic(self):
        """Higher ability → higher probability."""
        item = IRTItem(item_id="t1", difficulty=0.0, discrimination=1.5)
        probs = [item.probability(theta) for theta in [-2, -1, 0, 1, 2]]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_discrimination_effect(self):
        """Higher discrimination → steeper ICC."""
        item_low = IRTItem(item_id="t1", difficulty=0.0, discrimination=0.5)
        item_high = IRTItem(item_id="t2", difficulty=0.0, discrimination=2.0)
        # At θ = 1 (above difficulty), high-disc should have higher P
        assert item_high.probability(1.0) > item_low.probability(1.0)

    def test_guessing_parameter(self):
        """3PL: P should never go below guessing parameter."""
        item = IRTItem(item_id="t1", difficulty=0.0, discrimination=1.0, guessing=0.25)
        # Even at very low ability, P ≥ c
        assert item.probability(-10.0) >= 0.24

    def test_information_peak(self):
        """Information should peak near item difficulty."""
        item = IRTItem(item_id="t1", difficulty=1.0, discrimination=1.5)
        infos = [(theta, item.information(theta)) for theta in np.linspace(-3, 5, 50)]
        peak_theta = max(infos, key=lambda x: x[1])[0]
        assert abs(peak_theta - 1.0) < 0.5  # peak near b=1.0

    def test_information_nonnegative(self):
        item = IRTItem(item_id="t1", difficulty=0.0, discrimination=1.0)
        for theta in np.linspace(-4, 4, 20):
            assert item.information(theta) >= 0


# ============================================================================
# IRTModel
# ============================================================================

class TestIRTModel:
    @pytest.fixture
    def model_and_items(self):
        items = generate_item_bank(n_items=30, n_areas=3, seed=42)
        model = IRTModel(items)
        return model, items

    def test_mle_no_responses(self, model_and_items):
        model, _ = model_and_items
        est = model.estimate_ability_mle([])
        assert est.theta == 0.0
        assert est.n_items == 0

    def test_mle_all_correct(self, model_and_items):
        model, items = model_and_items
        # All correct → high estimated ability
        responses = [
            IRTResponse(learner_id="s1", item_id=items[i].item_id, correct=True)
            for i in range(10)
        ]
        est = model.estimate_ability_mle(responses)
        assert est.theta > 0.5

    def test_mle_all_incorrect(self, model_and_items):
        model, items = model_and_items
        responses = [
            IRTResponse(learner_id="s1", item_id=items[i].item_id, correct=False)
            for i in range(10)
        ]
        est = model.estimate_ability_mle(responses)
        assert est.theta < -0.5

    def test_eap_estimation(self, model_and_items):
        model, items = model_and_items
        # Mixed responses
        rng = np.random.RandomState(42)
        true_theta = 1.0
        responses = []
        for i in range(15):
            p = items[i].probability(true_theta)
            correct = rng.rand() < p
            responses.append(IRTResponse(
                learner_id="s1", item_id=items[i].item_id, correct=correct,
            ))

        est = model.estimate_ability_eap(responses)
        # Should be roughly near true_theta
        assert abs(est.theta - true_theta) < 1.5

    def test_eap_se_decreases_with_items(self, model_and_items):
        model, items = model_and_items
        rng = np.random.RandomState(42)
        responses = []
        ses = []
        for i in range(20):
            responses.append(IRTResponse(
                learner_id="s1", item_id=items[i].item_id,
                correct=rng.rand() < 0.6,
            ))
            est = model.estimate_ability_eap(responses)
            ses.append(est.se)

        # SE should generally decrease as more items are administered
        assert ses[-1] < ses[0]

    def test_test_information(self, model_and_items):
        model, _ = model_and_items
        info = model.test_information(0.0)
        assert info > 0


# ============================================================================
# Adaptive Selection
# ============================================================================

class TestAdaptiveSelection:
    @pytest.fixture
    def selector(self):
        items = generate_item_bank(n_items=40, n_areas=3, seed=42)
        model = IRTModel(items)
        return AdaptiveSelector(
            model=model,
            item_pool=items,
            criterion="mfi",
            max_items=15,
            se_threshold=0.3,
        )

    def test_start_session(self, selector):
        session = selector.start_session("test_learner")
        assert session.learner_id == "test_learner"
        assert session.n_administered == 0
        assert not session.terminated

    def test_select_next_returns_item(self, selector):
        session = selector.start_session("test_learner")
        result = selector.select_next(session)
        assert result is not None
        assert result.selected_item is not None
        assert result.criterion_value > 0

    def test_record_response_updates_estimate(self, selector):
        session = selector.start_session("test_learner")
        result = selector.select_next(session)
        est = selector.record_response(session, result.selected_item.item_id, True)
        assert est.n_items == 1
        assert session.n_administered == 1

    def test_no_item_reuse(self, selector):
        session = selector.start_session("test_learner")
        used_items = set()
        for _ in range(10):
            result = selector.select_next(session)
            if result is None:
                break
            assert result.selected_item.item_id not in used_items
            used_items.add(result.selected_item.item_id)
            selector.record_response(session, result.selected_item.item_id, True)

    def test_termination_max_items(self):
        items = generate_item_bank(n_items=40, seed=42)
        model = IRTModel(items)
        selector = AdaptiveSelector(
            model=model, item_pool=items,
            max_items=5, se_threshold=0.01,  # very tight SE → won't trigger
        )
        session = selector.start_session("test")
        for _ in range(10):
            result = selector.select_next(session)
            if result is None:
                break
            selector.record_response(session, result.selected_item.item_id, True)

        assert session.terminated
        assert session.termination_reason == "max_items"
        assert session.n_administered == 5


# ============================================================================
# CAT Simulation
# ============================================================================

class TestCATSimulation:
    def test_simulation_completes(self):
        items = generate_item_bank(n_items=50, seed=42)
        model = IRTModel(items)

        session = run_cat_simulation(
            model=model, item_pool=items,
            true_theta=1.0, max_items=20, se_threshold=0.3,
        )
        assert session.terminated
        assert session.n_administered > 0
        assert session.n_administered <= 20

    def test_estimate_accuracy(self):
        """CAT estimate should be reasonably close to true θ."""
        items = generate_item_bank(n_items=50, seed=42)
        model = IRTModel(items)

        true_theta = 0.5
        session = run_cat_simulation(
            model=model, item_pool=items,
            true_theta=true_theta, max_items=30, se_threshold=0.25,
        )
        est = session.current_estimate
        assert abs(est.theta - true_theta) < 1.5  # generous bound

    def test_different_abilities(self):
        """CAT should give different estimates for different abilities."""
        items = generate_item_bank(n_items=50, seed=42)
        model = IRTModel(items)

        session_low = run_cat_simulation(
            model=model, item_pool=items,
            true_theta=-1.5, max_items=20, seed=1,
        )
        session_high = run_cat_simulation(
            model=model, item_pool=items,
            true_theta=1.5, max_items=20, seed=2,
        )
        assert session_high.current_estimate.theta > session_low.current_estimate.theta
