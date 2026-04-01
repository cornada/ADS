"""Prerequisite DAG construction and analysis.

Builds directed acyclic graphs from course prerequisite data.
Each edge (A → B) means "A is a prerequisite for B".

Key analyses:
- Critical path: longest prerequisite chain (earliest to most advanced)
- Bottleneck detection: courses with highest betweenness centrality
- Topological layers: semester-like ordering respecting prerequisites
- DAG vs embedding distance: find hidden connections

Requires: networkx (already in stdlib-like usage across scientific Python)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class DAGStats:
    """Summary statistics for a prerequisite DAG."""

    n_nodes: int
    n_edges: int
    n_roots: int  # courses with no prerequisites
    n_leaves: int  # courses that are not prerequisites for anything
    n_components: int  # weakly connected components
    longest_path_length: int
    longest_path: List[str]
    mean_in_degree: float
    mean_out_degree: float
    max_in_degree: int
    max_out_degree: int
    max_in_degree_node: str
    max_out_degree_node: str
    density: float
    is_dag: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_roots": self.n_roots,
            "n_leaves": self.n_leaves,
            "n_components": self.n_components,
            "longest_path_length": self.longest_path_length,
            "longest_path": self.longest_path,
            "mean_in_degree": round(self.mean_in_degree, 2),
            "mean_out_degree": round(self.mean_out_degree, 2),
            "max_in_degree": self.max_in_degree,
            "max_out_degree": self.max_out_degree,
            "max_in_degree_node": self.max_in_degree_node,
            "max_out_degree_node": self.max_out_degree_node,
            "density": round(self.density, 6),
            "is_dag": self.is_dag,
        }


@dataclass
class BottleneckResult:
    """Bottleneck analysis result."""

    node: str
    betweenness: float
    in_degree: int
    out_degree: int
    descendants: int  # how many courses depend on this (transitively)
    ancestors: int  # how many courses precede this (transitively)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "betweenness": round(self.betweenness, 6),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "descendants": self.descendants,
            "ancestors": self.ancestors,
        }


@dataclass
class CriticalPath:
    """A critical (longest) path through the DAG."""

    path: List[str]
    length: int
    fgos_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "length": self.length,
            "fgos_code": self.fgos_code,
        }


@dataclass
class HiddenConnection:
    """A pair of courses far in DAG but close in embedding space."""

    course_a: str
    course_b: str
    dag_distance: int  # shortest path in DAG (or -1 if unreachable)
    embedding_distance: float  # cosine distance
    connection_type: str  # "far_dag_close_embed" or "close_dag_far_embed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_a": self.course_a,
            "course_b": self.course_b,
            "dag_distance": self.dag_distance,
            "embedding_distance": round(self.embedding_distance, 4),
            "connection_type": self.connection_type,
        }


def build_dag(
    df: pd.DataFrame,
    course_col: str = "course_name",
    prereq_col: str = "prerequisite_name",
    fgos_col: Optional[str] = "fgos_code",
    year: Optional[int] = None,
    fgos_filter: Optional[str] = None,
) -> nx.DiGraph:
    """Build prerequisite DAG from dataframe.

    Edge direction: prerequisite → course (A must come before B).

    Args:
        df: DataFrame with prerequisite edges
        course_col: column name for the course
        prereq_col: column name for the prerequisite
        fgos_col: optional column for FGOS code (stored as node attribute)
        year: filter to specific year
        fgos_filter: filter to specific FGOS code

    Returns:
        NetworkX DiGraph (may contain cycles — caller should check)
    """
    filtered = df.copy()
    if year is not None and "year" in filtered.columns:
        filtered = filtered[filtered["year"] == year]
    if fgos_filter is not None and fgos_col in filtered.columns:
        filtered = filtered[filtered[fgos_col] == fgos_filter]

    G = nx.DiGraph()

    for _, row in filtered.iterrows():
        course = str(row[course_col]).strip()
        prereq = str(row[prereq_col]).strip()
        if not course or not prereq or course == prereq:
            continue

        # Edge: prerequisite → course
        G.add_edge(prereq, course)

        # Store FGOS as node attribute
        if fgos_col and fgos_col in row.index:
            fgos = str(row[fgos_col])
            G.nodes[course]["fgos_code"] = fgos
            G.nodes[prereq]["fgos_code"] = G.nodes[prereq].get("fgos_code", fgos)

    return G


def ensure_dag(G: nx.DiGraph) -> nx.DiGraph:
    """Remove cycles to ensure the graph is a DAG.

    Uses feedback arc set heuristic to remove minimum edges.

    Returns:
        Copy of G with cycles removed
    """
    if nx.is_directed_acyclic_graph(G):
        return G.copy()

    H = G.copy()
    # Iteratively remove back-edges found by DFS
    while not nx.is_directed_acyclic_graph(H):
        try:
            cycle = nx.find_cycle(H, orientation="original")
            # Remove the last edge in the cycle (heuristic)
            u, v, _ = cycle[-1]
            H.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break

    return H


def compute_dag_stats(G: nx.DiGraph) -> DAGStats:
    """Compute comprehensive statistics for a DAG."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return DAGStats(
            n_nodes=0, n_edges=0, n_roots=0, n_leaves=0,
            n_components=0, longest_path_length=0, longest_path=[],
            mean_in_degree=0, mean_out_degree=0,
            max_in_degree=0, max_out_degree=0,
            max_in_degree_node="", max_out_degree_node="",
            density=0, is_dag=True,
        )

    is_dag = nx.is_directed_acyclic_graph(G)

    # Roots (no incoming edges) and leaves (no outgoing edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]

    # Components
    n_components = nx.number_weakly_connected_components(G)

    # Degree stats
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    mean_in = np.mean(list(in_degrees.values()))
    mean_out = np.mean(list(out_degrees.values()))
    max_in_node = max(in_degrees, key=in_degrees.get)
    max_out_node = max(out_degrees, key=out_degrees.get)

    # Longest path (only if DAG)
    if is_dag:
        longest = nx.dag_longest_path(G)
        longest_len = len(longest) - 1 if longest else 0
    else:
        longest = []
        longest_len = 0

    density = nx.density(G)

    return DAGStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_roots=len(roots),
        n_leaves=len(leaves),
        n_components=n_components,
        longest_path_length=longest_len,
        longest_path=longest,
        mean_in_degree=mean_in,
        mean_out_degree=mean_out,
        max_in_degree=in_degrees[max_in_node],
        max_out_degree=out_degrees[max_out_node],
        max_in_degree_node=max_in_node,
        max_out_degree_node=max_out_node,
        density=density,
        is_dag=is_dag,
    )


def find_bottlenecks(G: nx.DiGraph, top_k: int = 20) -> List[BottleneckResult]:
    """Find bottleneck courses — those with highest betweenness centrality.

    Bottleneck = a course that many prerequisite chains pass through.
    Removing it would break the most learning paths.

    Args:
        G: Prerequisite DAG
        top_k: number of top bottlenecks to return

    Returns:
        List of BottleneckResult sorted by betweenness (descending)
    """
    if G.number_of_nodes() == 0:
        return []

    betweenness = nx.betweenness_centrality(G)

    results = []
    for node, bc in sorted(betweenness.items(), key=lambda x: -x[1])[:top_k]:
        desc = nx.descendants(G, node) if G.has_node(node) else set()
        anc = nx.ancestors(G, node) if G.has_node(node) else set()
        results.append(BottleneckResult(
            node=node,
            betweenness=bc,
            in_degree=G.in_degree(node),
            out_degree=G.out_degree(node),
            descendants=len(desc),
            ancestors=len(anc),
        ))

    return results


def find_critical_paths(
    G: nx.DiGraph,
    top_k: int = 10,
    fgos_code: Optional[str] = None,
) -> List[CriticalPath]:
    """Find the longest paths (critical paths) in the DAG.

    In curriculum context: the longest prerequisite chain a student must
    complete before reaching the most advanced course.

    Args:
        G: Prerequisite DAG (must be acyclic)
        top_k: number of longest paths to return
        fgos_code: optional FGOS label for the result

    Returns:
        List of CriticalPath sorted by length (descending)
    """
    if not nx.is_directed_acyclic_graph(G) or G.number_of_nodes() == 0:
        return []

    # Find longest path from each root
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]

    paths = []
    for root in roots:
        for leaf in leaves:
            try:
                # All simple paths (limited to avoid combinatorial explosion)
                for path in nx.all_simple_paths(G, root, leaf):
                    paths.append(path)
                    if len(paths) > top_k * 100:
                        break
            except nx.NetworkXError:
                continue
            if len(paths) > top_k * 100:
                break

    # Sort by length, deduplicate
    paths.sort(key=len, reverse=True)
    seen = set()
    results = []
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            results.append(CriticalPath(
                path=p,
                length=len(p) - 1,
                fgos_code=fgos_code,
            ))
        if len(results) >= top_k:
            break

    return results


def topological_layers(G: nx.DiGraph) -> List[List[str]]:
    """Compute topological layers (semester-like ordering).

    Layer 0: courses with no prerequisites (roots)
    Layer 1: courses whose prerequisites are all in layer 0
    Layer k: courses whose prerequisites are all in layers 0..k-1

    Args:
        G: Prerequisite DAG

    Returns:
        List of layers, each a list of course names
    """
    if not nx.is_directed_acyclic_graph(G):
        return []

    layers = []
    remaining = set(G.nodes())
    completed = set()

    while remaining:
        # Current layer: nodes whose all predecessors are completed
        layer = []
        for node in remaining:
            preds = set(G.predecessors(node))
            if preds.issubset(completed):
                layer.append(node)

        if not layer:
            # Stuck — shouldn't happen in a DAG, but safety check
            break

        layers.append(sorted(layer))
        completed.update(layer)
        remaining -= set(layer)

    return layers


def compare_dag_vs_embedding(
    G: nx.DiGraph,
    embeddings: Dict[str, np.ndarray],
    sample_size: int = 2000,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[HiddenConnection]]:
    """Compare DAG distance vs embedding distance for node pairs.

    Finds "hidden connections": courses that are far in DAG topology
    but close in embedding space (similar content, different prerequisites).

    Args:
        G: Prerequisite DAG
        embeddings: dict mapping node name → embedding vector
        sample_size: number of pairs to sample
        rng_seed: random seed

    Returns:
        Tuple of (dag_distances, embedding_distances, hidden_connections)
    """
    # Get nodes with embeddings
    nodes_with_emb = [n for n in G.nodes() if n in embeddings]
    if len(nodes_with_emb) < 10:
        return np.array([]), np.array([]), []

    rng = np.random.RandomState(rng_seed)

    # Convert DAG to undirected for shortest path computation
    G_undirected = G.to_undirected()

    # Sample pairs
    n = len(nodes_with_emb)
    n_pairs = min(sample_size, n * (n - 1) // 2)
    pairs_idx = set()
    while len(pairs_idx) < n_pairs:
        i, j = rng.choice(n, size=2, replace=False)
        pair = (min(i, j), max(i, j))
        pairs_idx.add(pair)

    dag_dists = []
    emb_dists = []
    hidden = []

    for i, j in pairs_idx:
        na, nb = nodes_with_emb[i], nodes_with_emb[j]

        # DAG distance (undirected shortest path)
        try:
            d_dag = nx.shortest_path_length(G_undirected, na, nb)
        except nx.NetworkXNoPath:
            d_dag = -1  # unreachable

        # Embedding distance (cosine)
        ea, eb = embeddings[na], embeddings[nb]
        cos_sim = np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8)
        d_emb = 1.0 - cos_sim

        dag_dists.append(d_dag)
        emb_dists.append(d_emb)

        # Hidden connections: far in DAG (≥4 hops) but close in embedding (<0.3)
        if d_dag >= 4 and d_emb < 0.3:
            hidden.append(HiddenConnection(
                course_a=na, course_b=nb,
                dag_distance=d_dag, embedding_distance=d_emb,
                connection_type="far_dag_close_embed",
            ))
        # Reverse: close in DAG (=1 direct prereq) but far in embedding (>0.7)
        elif d_dag == 1 and d_emb > 0.7:
            hidden.append(HiddenConnection(
                course_a=na, course_b=nb,
                dag_distance=d_dag, embedding_distance=d_emb,
                connection_type="close_dag_far_embed",
            ))

    return np.array(dag_dists), np.array(emb_dists), hidden


def per_fgos_dags(
    df: pd.DataFrame,
    year: Optional[int] = None,
) -> Dict[str, nx.DiGraph]:
    """Build separate DAGs for each FGOS direction.

    Args:
        df: Prerequisites dataframe
        year: optional year filter

    Returns:
        Dict mapping FGOS code → DAG
    """
    filtered = df.copy()
    if year is not None and "year" in filtered.columns:
        filtered = filtered[filtered["year"] == year]

    dags = {}
    for fgos_code in filtered["fgos_code"].unique():
        G = build_dag(filtered, fgos_filter=fgos_code)
        if G.number_of_nodes() > 0:
            G = ensure_dag(G)
            dags[fgos_code] = G

    return dags
