"""Causal DAG specification for curriculum → outcome relationships.

Defines the assumed causal structure linking curriculum design decisions
to observable outcomes. This is *not* causal inference from data — it is
a formal declaration of assumptions that makes our correlational analysis
transparent and auditable.

Key DAG structure:
    FGOS_Policy → Curriculum_Structure → Student_Competencies → Market_Fit
                  ↑                      ↑
              University_Resources   Student_Ability (unobserved)

Nodes:
- FGOS_Policy: Federal education standard version (observed, exogenous)
- Curriculum_Structure: courses, credits, prerequisites (observed)
- Embedding_Position: position in didactic space (observed, computed)
- Competency_Coverage: fraction of target competencies covered (observed)
- Market_Fit: alignment with labor market (observed)
- University_Resources: faculty, funding, infrastructure (partially observed)
- Student_Ability: latent talent/motivation (unobserved confounder)
- Regional_Economy: local job market conditions (partially observed)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


class NodeType(Enum):
    OBSERVED = "observed"
    PARTIALLY_OBSERVED = "partially_observed"
    UNOBSERVED = "unobserved"
    INSTRUMENT = "instrument"  # exogenous shock


@dataclass(frozen=True)
class CausalNode:
    """A node in the causal DAG."""

    name: str
    node_type: NodeType
    description: str = ""


@dataclass(frozen=True)
class CausalEdge:
    """A directed edge in the causal DAG."""

    source: str
    target: str
    mechanism: str = ""  # how source affects target


@dataclass
class CausalDAG:
    """Causal DAG for curriculum → outcome analysis.

    This DAG encodes assumptions, not discoveries. It makes explicit
    which confounders we worry about and what we assume is exogenous.
    """

    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: CausalNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, edge: CausalEdge) -> None:
        self.edges.append(edge)

    def parents(self, node_name: str) -> List[str]:
        return [e.source for e in self.edges if e.target == node_name]

    def children(self, node_name: str) -> List[str]:
        return [e.target for e in self.edges if e.source == node_name]

    def unobserved_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Find unobserved nodes that could confound treatment → outcome."""
        confounders = []
        for name, node in self.nodes.items():
            if node.node_type in (NodeType.UNOBSERVED, NodeType.PARTIALLY_OBSERVED):
                # Check if this node has paths to both treatment and outcome
                t_children = self.children(name)
                if self._has_path_to(name, treatment) and self._has_path_to(name, outcome):
                    confounders.append(name)
        return confounders

    def _has_path_to(self, source: str, target: str) -> bool:
        """Check if there's a directed path from source to target (BFS)."""
        visited: Set[str] = set()
        queue = [source]
        while queue:
            current = queue.pop(0)
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.children(current))
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {
                name: {"type": n.node_type.value, "description": n.description}
                for name, n in self.nodes.items()
            },
            "edges": [
                {"source": e.source, "target": e.target, "mechanism": e.mechanism}
                for e in self.edges
            ],
            "metadata": self.metadata,
        }


def build_ads_causal_dag() -> CausalDAG:
    """Build the standard ADS causal DAG for curriculum analysis.

    This encodes our domain knowledge about what causes what in the
    education → market pipeline.
    """
    dag = CausalDAG(metadata={"version": "1.0", "domain": "curriculum_analysis"})

    # Nodes
    dag.add_node(CausalNode(
        "fgos_policy", NodeType.INSTRUMENT,
        "Federal education standard version (exogenous policy shock)",
    ))
    dag.add_node(CausalNode(
        "curriculum_structure", NodeType.OBSERVED,
        "Courses, credits, prerequisites, competency mappings",
    ))
    dag.add_node(CausalNode(
        "embedding_position", NodeType.OBSERVED,
        "Course position in didactic embedding space (384d)",
    ))
    dag.add_node(CausalNode(
        "competency_coverage", NodeType.OBSERVED,
        "Fraction of target competencies covered by curriculum",
    ))
    dag.add_node(CausalNode(
        "market_fit", NodeType.OBSERVED,
        "Cosine similarity to labor market embeddings",
    ))
    dag.add_node(CausalNode(
        "university_resources", NodeType.PARTIALLY_OBSERVED,
        "Faculty quality, funding, infrastructure (partially observed via proxy)",
    ))
    dag.add_node(CausalNode(
        "student_ability", NodeType.UNOBSERVED,
        "Latent student talent and motivation",
    ))
    dag.add_node(CausalNode(
        "regional_economy", NodeType.PARTIALLY_OBSERVED,
        "Local job market conditions affecting both curriculum and outcomes",
    ))

    # Edges (assumed causal structure)
    dag.add_edge(CausalEdge(
        "fgos_policy", "curriculum_structure",
        "Policy mandates course requirements, competency standards",
    ))
    dag.add_edge(CausalEdge(
        "curriculum_structure", "embedding_position",
        "Course content determines embedding coordinates",
    ))
    dag.add_edge(CausalEdge(
        "curriculum_structure", "competency_coverage",
        "Curriculum design determines competency coverage",
    ))
    dag.add_edge(CausalEdge(
        "embedding_position", "market_fit",
        "Embedding proximity to job embeddings = market alignment",
    ))
    dag.add_edge(CausalEdge(
        "competency_coverage", "market_fit",
        "Competency coverage affects employability",
    ))
    dag.add_edge(CausalEdge(
        "university_resources", "curriculum_structure",
        "Resources constrain what courses can be offered",
    ))
    dag.add_edge(CausalEdge(
        "university_resources", "market_fit",
        "University reputation affects placement (brand signal)",
    ))
    dag.add_edge(CausalEdge(
        "student_ability", "competency_coverage",
        "Able students acquire competencies faster",
    ))
    dag.add_edge(CausalEdge(
        "student_ability", "market_fit",
        "Talent affects outcomes regardless of curriculum",
    ))
    dag.add_edge(CausalEdge(
        "regional_economy", "curriculum_structure",
        "Local industry needs influence curriculum design",
    ))
    dag.add_edge(CausalEdge(
        "regional_economy", "market_fit",
        "Local economy affects employment outcomes directly",
    ))

    return dag
