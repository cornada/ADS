"""Tests for prerequisite DAG module."""
import numpy as np
import pandas as pd
import pytest

from ads_core.graph.prerequisite_dag import (
    build_dag,
    ensure_dag,
    compute_dag_stats,
    find_bottlenecks,
    find_critical_paths,
    topological_layers,
    compare_dag_vs_embedding,
    per_fgos_dags,
)


@pytest.fixture
def simple_prereqs():
    """Linear chain: A → B → C → D with branch A → E → D."""
    return pd.DataFrame({
        "course_name": ["B", "C", "D", "E", "D"],
        "prerequisite_name": ["A", "B", "C", "A", "E"],
        "fgos_code": ["F1", "F1", "F1", "F1", "F1"],
        "year": [2024] * 5,
    })


@pytest.fixture
def multi_fgos_prereqs():
    """Two FGOS: F1 has A→B→C, F2 has X→Y."""
    return pd.DataFrame({
        "course_name": ["B", "C", "Y"],
        "prerequisite_name": ["A", "B", "X"],
        "fgos_code": ["F1", "F1", "F2"],
        "year": [2024, 2024, 2024],
    })


class TestBuildDAG:
    def test_basic(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        assert G.number_of_nodes() == 5  # A, B, C, D, E
        assert G.number_of_edges() == 5

    def test_edge_direction(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        assert G.has_edge("A", "B")  # prereq → course
        assert not G.has_edge("B", "A")

    def test_year_filter(self, simple_prereqs):
        G = build_dag(simple_prereqs, year=2025)
        assert G.number_of_nodes() == 0  # no 2025 data

    def test_fgos_filter(self, multi_fgos_prereqs):
        G = build_dag(multi_fgos_prereqs, fgos_filter="F1")
        assert G.number_of_nodes() == 3  # A, B, C
        assert not G.has_node("X")

    def test_removes_self_loops(self):
        df = pd.DataFrame({
            "course_name": ["A", "B"],
            "prerequisite_name": ["A", "A"],
            "fgos_code": ["F1", "F1"],
            "year": [2024, 2024],
        })
        G = build_dag(df)
        assert not G.has_edge("A", "A")
        assert G.has_edge("A", "B")


class TestEnsureDAG:
    def test_already_dag(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        H = ensure_dag(G)
        assert H.number_of_edges() == G.number_of_edges()

    def test_removes_cycles(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        H = ensure_dag(G)
        assert nx.is_directed_acyclic_graph(H)
        assert H.number_of_edges() == 2  # one edge removed


class TestDAGStats:
    def test_basic_stats(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        stats = compute_dag_stats(G)
        assert stats.n_nodes == 5
        assert stats.n_edges == 5
        assert stats.is_dag
        assert stats.n_roots == 1  # A
        assert stats.n_leaves == 1  # D
        assert stats.longest_path_length >= 3  # A → B → C → D

    def test_empty_graph(self):
        import networkx as nx
        G = nx.DiGraph()
        stats = compute_dag_stats(G)
        assert stats.n_nodes == 0
        assert stats.is_dag


class TestBottlenecks:
    def test_finds_bottleneck(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        bottlenecks = find_bottlenecks(G, top_k=5)
        assert len(bottlenecks) > 0
        # B or C should have highest betweenness (on the main chain)
        top_nodes = {b.node for b in bottlenecks[:2]}
        assert "A" in top_nodes or "B" in top_nodes or "C" in top_nodes


class TestCriticalPaths:
    def test_finds_longest(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        paths = find_critical_paths(G, top_k=3)
        assert len(paths) > 0
        assert paths[0].length >= 3  # A → B → C → D

    def test_path_is_valid(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        paths = find_critical_paths(G, top_k=1)
        path = paths[0].path
        for i in range(len(path) - 1):
            assert G.has_edge(path[i], path[i + 1])


class TestTopologicalLayers:
    def test_layers(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        layers = topological_layers(G)
        assert len(layers) >= 2
        # A should be in layer 0 (root)
        assert "A" in layers[0]
        # D should be in the last layer
        assert "D" in layers[-1]

    def test_all_nodes_assigned(self, simple_prereqs):
        G = build_dag(simple_prereqs)
        layers = topological_layers(G)
        all_nodes = set()
        for layer in layers:
            all_nodes.update(layer)
        assert all_nodes == set(G.nodes())


class TestDAGvsEmbedding:
    def test_basic(self):
        """Build a larger graph (>10 nodes) for embedding comparison."""
        import networkx as nx
        G = nx.DiGraph()
        # Chain of 15 nodes
        for i in range(14):
            G.add_edge(f"N{i}", f"N{i+1}")
        rng = np.random.RandomState(42)
        embeddings = {n: rng.randn(64).astype(np.float32) for n in G.nodes()}
        dag_d, emb_d, hidden = compare_dag_vs_embedding(G, embeddings, sample_size=20)
        assert len(dag_d) == len(emb_d)
        assert len(dag_d) > 0


class TestPerFGOSDAGs:
    def test_splits_by_fgos(self, multi_fgos_prereqs):
        dags = per_fgos_dags(multi_fgos_prereqs)
        assert "F1" in dags
        assert "F2" in dags
        assert dags["F1"].number_of_nodes() == 3
        assert dags["F2"].number_of_nodes() == 2
