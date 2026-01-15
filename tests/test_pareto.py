"""Tests for Pareto front computation."""
from ads_core.eval.pareto import pareto_front, dominates, pareto_rank


class TestDominates:
    """Tests for dominates function."""

    def test_dominates_better_both(self):
        """a dominates b if a is better in all dimensions."""
        a = {"x": 2.0, "y": 3.0}
        b = {"x": 1.0, "y": 2.0}
        assert dominates(a, b, ["x", "y"]) is True

    def test_dominates_equal_one_better_other(self):
        """a dominates b if equal in one, better in other."""
        a = {"x": 2.0, "y": 3.0}
        b = {"x": 2.0, "y": 2.0}
        assert dominates(a, b, ["x", "y"]) is True

    def test_not_dominates_tradeoff(self):
        """Neither dominates when there's a trade-off."""
        a = {"x": 2.0, "y": 1.0}
        b = {"x": 1.0, "y": 2.0}
        assert dominates(a, b, ["x", "y"]) is False
        assert dominates(b, a, ["x", "y"]) is False

    def test_not_dominates_equal(self):
        """Equal solutions don't dominate each other."""
        a = {"x": 1.0, "y": 1.0}
        b = {"x": 1.0, "y": 1.0}
        assert dominates(a, b, ["x", "y"]) is False


class TestParetoFront:
    """Tests for pareto_front function."""

    def test_pareto_front_single_dominator(self):
        """One solution dominates all others."""
        items = [
            {"a": 1.0, "b": 2.0},  # dominated by item 2
            {"a": 2.0, "b": 1.0},  # dominated by item 2
            {"a": 3.0, "b": 3.0},  # dominates all
        ]
        idx = pareto_front(items, keys=["a", "b"])
        assert idx == [2]

    def test_pareto_front_all_tradeoffs(self):
        """All solutions are trade-offs (all on front)."""
        items = [
            {"a": 3.0, "b": 1.0},  # best in a
            {"a": 1.0, "b": 3.0},  # best in b
            {"a": 2.0, "b": 2.0},  # middle
        ]
        idx = pareto_front(items, keys=["a", "b"])
        # All are non-dominated (trade-offs)
        assert set(idx) == {0, 1, 2}

    def test_pareto_front_two_on_front(self):
        """Two solutions on front, one dominated."""
        items = [
            {"a": 3.0, "b": 1.0},  # on front
            {"a": 1.0, "b": 3.0},  # on front
            {"a": 1.0, "b": 1.0},  # dominated by both
        ]
        idx = pareto_front(items, keys=["a", "b"])
        assert set(idx) == {0, 1}

    def test_pareto_front_empty(self):
        """Empty input returns empty."""
        idx = pareto_front([], keys=["a", "b"])
        assert idx == []

    def test_pareto_front_single(self):
        """Single item is always on front."""
        items = [{"a": 1.0, "b": 2.0}]
        idx = pareto_front(items, keys=["a", "b"])
        assert idx == [0]


class TestParetoRank:
    """Tests for pareto_rank function."""

    def test_pareto_rank_layered(self):
        """Test ranking of layered solutions."""
        items = [
            {"a": 3.0, "b": 3.0},  # rank 0 (best)
            {"a": 2.0, "b": 2.0},  # rank 1
            {"a": 1.0, "b": 1.0},  # rank 2 (worst)
        ]
        ranks = pareto_rank(items, keys=["a", "b"])
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert ranks[2] == 2

    def test_pareto_rank_all_front(self):
        """All trade-offs have rank 0."""
        items = [
            {"a": 3.0, "b": 1.0},
            {"a": 1.0, "b": 3.0},
        ]
        ranks = pareto_rank(items, keys=["a", "b"])
        assert ranks == [0, 0]

    def test_pareto_rank_empty(self):
        """Empty input returns empty."""
        ranks = pareto_rank([], keys=["a", "b"])
        assert ranks == []
