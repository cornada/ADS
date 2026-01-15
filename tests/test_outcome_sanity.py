"""Tests for outcome sanity checking modules."""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from ads_core.outcomes.normalize import (
    CanonicalOutcome,
    normalize_ucb_outcomes,
    normalize_asu_outcomes,
    load_normalized_outcomes,
    outcomes_to_dataframe,
)
from ads_core.market.targets import (
    MarketCategory,
    MARKET_CATEGORIES,
    get_market_category_texts,
    create_market_target_vectors,
    compute_market_similarity,
    predict_market_distribution,
    keyword_baseline_prediction,
)
from ads_core.analysis.outcome_sanity import (
    OutcomeSanityResult,
    UnitSanityResult,
    compute_outcome_sanity,
    compute_keyword_baseline,
    aggregate_sanity_results,
)


class TestCanonicalOutcome:
    """Tests for CanonicalOutcome dataclass."""

    def test_create_outcome(self):
        """Create canonical outcome."""
        outcome = CanonicalOutcome(
            institution="UCB",
            unit="COE:Computer Science",
            unit_name="Computer Science (College of Engineering)",
            cohort_year="2024",
            employment_rate=0.85,
            grad_school_rate=0.12,
        )
        assert outcome.institution == "UCB"
        assert outcome.employment_rate == 0.85
        # career_outcomes_rate should be computed
        assert outcome.career_outcomes_rate == pytest.approx(0.97, rel=0.01)

    def test_compute_seeking_rate(self):
        """Seeking rate computed from career outcomes."""
        outcome = CanonicalOutcome(
            institution="ASU",
            unit="FSE",
            unit_name="Fulton Schools of Engineering",
            cohort_year="2015-2016",
            employment_rate=0.72,
            grad_school_rate=0.18,
        )
        # career_outcomes = 0.90, seeking = 0.10
        assert outcome.career_outcomes_rate == pytest.approx(0.90, rel=0.01)
        assert outcome.seeking_rate == pytest.approx(0.10, rel=0.01)

    def test_to_dict(self):
        """Convert outcome to dictionary."""
        outcome = CanonicalOutcome(
            institution="UCB",
            unit="L&S:Economics",
            unit_name="Economics",
            cohort_year="2024",
            employment_rate=0.78,
            median_salary=75000,
        )
        d = outcome.to_dict()
        assert d["institution"] == "UCB"
        assert d["employment_rate"] == 0.78
        assert d["median_salary"] == 75000


class TestOutcomesNormalization:
    """Tests for outcomes normalization functions."""

    def test_normalize_ucb_outcomes(self, tmp_path):
        """Normalize UCB outcomes from CSV."""
        # Create test CSV
        csv_content = """year,college,major,employment_rate,grad_school_rate,median_salary,n_respondents
2024,COE,Computer Science,0.85,0.12,130000,450
2024,L&S,Economics,0.78,0.15,75000,350
"""
        csv_path = tmp_path / "ucb_test.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        outcomes = normalize_ucb_outcomes(csv_path)
        assert len(outcomes) == 2

        cs_outcome = outcomes[0]
        assert cs_outcome.institution == "UCB"
        assert cs_outcome.unit == "COE:Computer Science"
        assert cs_outcome.employment_rate == 0.85
        assert cs_outcome.median_salary == 130000

    def test_normalize_asu_outcomes(self, tmp_path):
        """Normalize ASU outcomes from CSV."""
        csv_content = """college,college_name,academic_year,employment_rate,grad_school_rate,career_outcomes_rate,median_salary,n_respondents
FSE,Fulton Schools of Engineering,2015-2016,0.72,0.18,0.90,62000,1850
WPC,W. P. Carey School of Business,2015-2016,0.78,0.08,0.86,52000,1540
"""
        csv_path = tmp_path / "asu_test.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        outcomes = normalize_asu_outcomes(csv_path)
        assert len(outcomes) == 2

        fse = outcomes[0]
        assert fse.institution == "ASU"
        assert fse.unit == "FSE"
        assert fse.career_outcomes_rate == 0.90

    def test_load_ucb_outcomes(self):
        """Load UCB outcomes from fixtures."""
        outcomes = load_normalized_outcomes("ucb")
        assert len(outcomes) > 0
        # All should be UCB
        assert all(o.institution == "UCB" for o in outcomes)
        # All should have employment rate
        assert all(o.employment_rate is not None for o in outcomes)

    def test_load_asu_outcomes(self):
        """Load ASU outcomes from fixtures."""
        outcomes = load_normalized_outcomes("asu")
        assert len(outcomes) > 0
        assert all(o.institution == "ASU" for o in outcomes)

    def test_outcomes_to_dataframe(self):
        """Convert outcomes to DataFrame."""
        outcomes = [
            CanonicalOutcome(
                institution="UCB",
                unit="test:1",
                unit_name="Test 1",
                cohort_year="2024",
                employment_rate=0.8,
            ),
            CanonicalOutcome(
                institution="UCB",
                unit="test:2",
                unit_name="Test 2",
                cohort_year="2024",
                employment_rate=0.9,
            ),
        ]
        df = outcomes_to_dataframe(outcomes)
        assert len(df) == 2
        assert "employment_rate" in df.columns
        assert df["employment_rate"].mean() == pytest.approx(0.85)


class TestMarketCategories:
    """Tests for market category definitions."""

    def test_market_categories_defined(self):
        """Verify market categories are defined."""
        assert len(MARKET_CATEGORIES) >= 10
        assert "software_tech" in MARKET_CATEGORIES
        assert "data_ml" in MARKET_CATEGORIES
        assert "finance_business" in MARKET_CATEGORIES

    def test_market_category_structure(self):
        """Verify market category has required fields."""
        cat = MARKET_CATEGORIES["software_tech"]
        assert cat.category_id == "software_tech"
        assert cat.name
        assert cat.description
        assert len(cat.keywords) > 0
        assert len(cat.onet_groups) > 0

    def test_get_market_category_texts(self):
        """Get text representations."""
        texts = get_market_category_texts()
        assert len(texts) == len(MARKET_CATEGORIES)
        assert "software_tech" in texts
        assert "Software" in texts["software_tech"]

    def test_create_market_target_vectors(self):
        """Create market target vectors with encoder."""
        # Use simple mock encoder
        def mock_encoder(texts):
            return np.random.randn(len(texts), 64)

        vectors = create_market_target_vectors(mock_encoder)
        assert len(vectors) == len(MARKET_CATEGORIES)
        assert all(v.shape == (64,) for v in vectors.values())

    def test_compute_market_similarity(self):
        """Compute similarity to market categories."""
        # Create simple vectors
        unit_vec = np.array([1.0, 0.0, 0.0])
        market_vectors = {
            "cat1": np.array([1.0, 0.0, 0.0]),
            "cat2": np.array([0.0, 1.0, 0.0]),
        }

        sims = compute_market_similarity(unit_vec, market_vectors)
        assert "cat1" in sims
        assert "cat2" in sims
        assert sims["cat1"] > sims["cat2"]  # Should be more similar to cat1

    def test_predict_market_distribution(self):
        """Predict distribution over categories."""
        unit_vec = np.random.randn(64)
        market_vectors = {
            cat_id: np.random.randn(64)
            for cat_id in MARKET_CATEGORIES.keys()
        }

        dist = predict_market_distribution(unit_vec, market_vectors)
        assert len(dist) == len(MARKET_CATEGORIES)
        assert abs(sum(dist.values()) - 1.0) < 0.01  # Should sum to ~1

    def test_keyword_baseline(self):
        """Test keyword baseline prediction."""
        # Computer Science should match software_tech
        pred = keyword_baseline_prediction("Computer Science")
        assert "software_tech" in pred
        assert pred["software_tech"] > 0

        # Economics should match finance
        pred = keyword_baseline_prediction("Economics")
        assert pred["finance_business"] > 0

        # Unknown major should give uniform distribution
        pred = keyword_baseline_prediction("Unknown Field XYZ")
        assert all(v > 0 for v in pred.values())


class TestOutcomeSanityAnalysis:
    """Tests for outcome sanity analysis."""

    @pytest.fixture
    def sample_outcomes(self):
        """Create sample outcomes for testing."""
        return [
            CanonicalOutcome(
                institution="UCB",
                unit="test:cs",
                unit_name="Computer Science",
                cohort_year="2024",
                employment_rate=0.85,
                median_salary=130000,
            ),
            CanonicalOutcome(
                institution="UCB",
                unit="test:econ",
                unit_name="Economics",
                cohort_year="2024",
                employment_rate=0.78,
                median_salary=75000,
            ),
            CanonicalOutcome(
                institution="UCB",
                unit="test:eng",
                unit_name="English",
                cohort_year="2024",
                employment_rate=0.50,
                median_salary=48000,
            ),
        ]

    @pytest.fixture
    def sample_vectors(self, sample_outcomes):
        """Create sample unit vectors."""
        np.random.seed(42)
        return {o.unit: np.random.randn(64) for o in sample_outcomes}

    @pytest.fixture
    def market_vectors(self):
        """Create sample market vectors."""
        np.random.seed(42)
        return {cat_id: np.random.randn(64) for cat_id in MARKET_CATEGORIES.keys()}

    def test_compute_outcome_sanity(self, sample_outcomes, sample_vectors, market_vectors):
        """Compute outcome sanity metrics."""
        result = compute_outcome_sanity(
            dataset_id="test",
            outcomes=sample_outcomes,
            unit_vectors=sample_vectors,
            market_vectors=market_vectors,
            method="test_embedding",
        )

        assert result.dataset_id == "test"
        assert result.n_units == 3
        assert result.n_with_outcomes == 3
        assert result.coverage == 1.0
        assert len(result.unit_results) == 3

    def test_compute_keyword_baseline(self, sample_outcomes, market_vectors):
        """Compute keyword baseline."""
        result = compute_keyword_baseline(
            dataset_id="test",
            outcomes=sample_outcomes,
            market_vectors=market_vectors,
        )

        assert result.method == "keyword_baseline"
        assert result.n_units == 3
        assert len(result.unit_results) == 3

        # Check that CS gets assigned to software
        cs_result = next(r for r in result.unit_results if "Computer" in r.unit_name)
        assert "software" in cs_result.predicted_top_category or "data" in cs_result.predicted_top_category

    def test_aggregate_sanity_results(self, sample_outcomes, sample_vectors, market_vectors):
        """Aggregate results across methods."""
        results = [
            compute_outcome_sanity("test", sample_outcomes, sample_vectors, market_vectors, "embedding"),
            compute_keyword_baseline("test", sample_outcomes, market_vectors),
        ]

        summary = aggregate_sanity_results(results)
        assert summary["n_datasets"] == 1
        assert len(summary["methods"]) == 2
        assert "by_method" in summary


class TestIntegration:
    """Integration tests for outcome sanity check."""

    def test_full_pipeline_ucb(self):
        """Run full pipeline on UCB fixtures."""
        from ads_core.embed.encoder import StubEncoder

        # Load outcomes
        outcomes = load_normalized_outcomes("ucb")
        assert len(outcomes) > 0

        # Create encoder
        encoder = StubEncoder(d=64)

        # Create market vectors
        market_texts = get_market_category_texts()
        market_embeddings = encoder.encode(list(market_texts.values()))
        market_vectors = {
            cat_id: market_embeddings[i]
            for i, cat_id in enumerate(market_texts.keys())
        }

        # Create unit vectors
        unit_names = list({o.unit_name for o in outcomes})
        unit_embeddings = encoder.encode(unit_names)
        unit_vectors = {}
        for o in outcomes:
            if o.unit_name in unit_names:
                idx = unit_names.index(o.unit_name)
                unit_vectors[o.unit] = unit_embeddings[idx]

        # Compute sanity
        result = compute_outcome_sanity(
            dataset_id="ucb",
            outcomes=outcomes,
            unit_vectors=unit_vectors,
            market_vectors=market_vectors,
            method="stub_embedding",
        )

        assert result.dataset_id == "ucb"
        assert result.coverage > 0.5
        assert len(result.unit_results) > 0

    def test_full_pipeline_asu(self):
        """Run full pipeline on ASU fixtures."""
        from ads_core.embed.encoder import StubEncoder

        outcomes = load_normalized_outcomes("asu")
        assert len(outcomes) > 0

        encoder = StubEncoder(d=64)

        market_texts = get_market_category_texts()
        market_embeddings = encoder.encode(list(market_texts.values()))
        market_vectors = {
            cat_id: market_embeddings[i]
            for i, cat_id in enumerate(market_texts.keys())
        }

        unit_names = list({o.unit_name for o in outcomes})
        unit_embeddings = encoder.encode(unit_names)
        unit_vectors = {}
        for o in outcomes:
            if o.unit_name in unit_names:
                idx = unit_names.index(o.unit_name)
                unit_vectors[o.unit] = unit_embeddings[idx]

        result = compute_outcome_sanity(
            dataset_id="asu",
            outcomes=outcomes,
            unit_vectors=unit_vectors,
            market_vectors=market_vectors,
            method="stub_embedding",
        )

        assert result.dataset_id == "asu"
        assert result.coverage > 0.5
