"""Prerequisite graph API endpoints.

Returns course prerequisite DAGs in Cytoscape.js JSON format,
filterable by FGOS program code and academic year.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREQ_PATH = _REPO_ROOT / "data" / "raw" / "misis" / "prerequisites.csv"


def _load_prerequisites() -> List[Dict[str, str]]:
    """Load prerequisite edges from CSV."""
    if not _PREREQ_PATH.exists():
        return []
    with _PREREQ_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@router.get("/programs")
def list_programs():
    """List available FGOS programs with edge counts."""
    rows = _load_prerequisites()
    programs: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r["fgos_code"]
        if key not in programs:
            programs[key] = {"fgos_code": key, "years": set(), "edge_count": 0}
        programs[key]["years"].add(r["year"])
        programs[key]["edge_count"] += 1

    result = []
    for p in sorted(programs.values(), key=lambda x: -x["edge_count"]):
        result.append({
            "fgos_code": p["fgos_code"],
            "years": sorted(p["years"]),
            "edge_count": p["edge_count"],
        })
    return result


@router.get("/prerequisites")
def get_prerequisite_graph(
    fgos_code: str = Query(..., description="FGOS program code (e.g. 09.03.01)"),
    year: Optional[str] = Query(None, description="Academic year filter (e.g. 2024)"),
):
    """Get prerequisite graph in Cytoscape.js JSON format.

    Returns nodes (courses) and edges (prerequisite relationships)
    for a specific FGOS program.
    """
    rows = _load_prerequisites()

    # Filter
    filtered = [r for r in rows if r["fgos_code"] == fgos_code]
    if year:
        filtered = [r for r in filtered if r["year"] == year]

    if not filtered:
        return {"elements": {"nodes": [], "edges": []}, "stats": {"nodes": 0, "edges": 0}}

    # Build node set and edge list
    nodes: Dict[str, Dict[str, Any]] = {}
    edges = []
    in_degree: Dict[str, int] = defaultdict(int)
    out_degree: Dict[str, int] = defaultdict(int)

    for r in filtered:
        src = r["prerequisite_name"]
        tgt = r["course_name"]

        if src not in nodes:
            nodes[src] = {"id": src, "label": src, "is_prereq_only": True}
        if tgt not in nodes:
            nodes[tgt] = {"id": tgt, "label": tgt, "is_prereq_only": False}
        # If a node appears as a target, it's not prereq-only
        nodes[tgt]["is_prereq_only"] = False

        edges.append({"source": src, "target": tgt})
        in_degree[tgt] += 1
        out_degree[src] += 1

    # Compute node properties for visualization
    cyto_nodes = []
    for nid, ndata in nodes.items():
        cyto_nodes.append({
            "data": {
                "id": nid,
                "label": nid,
                "in_degree": in_degree.get(nid, 0),
                "out_degree": out_degree.get(nid, 0),
                "is_root": in_degree.get(nid, 0) == 0,
                "is_leaf": out_degree.get(nid, 0) == 0,
            }
        })

    cyto_edges = []
    for e in edges:
        cyto_edges.append({
            "data": {
                "source": e["source"],
                "target": e["target"],
            }
        })

    return {
        "elements": {
            "nodes": cyto_nodes,
            "edges": cyto_edges,
        },
        "stats": {
            "nodes": len(cyto_nodes),
            "edges": len(cyto_edges),
            "roots": sum(1 for n in cyto_nodes if n["data"]["is_root"]),
            "leaves": sum(1 for n in cyto_nodes if n["data"]["is_leaf"]),
            "fgos_code": fgos_code,
            "year": year,
        },
    }
