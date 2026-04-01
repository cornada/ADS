"""Prerequisite graph API endpoints.

Returns course prerequisite DAGs in Cytoscape.js JSON format,
filterable by FGOS program code and academic year.
Also provides cross-program network, temporal diff, and aggregate stats.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data" / "raw" / "misis"
_PREREQ_PATH = _DATA_DIR / "prerequisites.csv"
_TEMPORAL_PATH = _DATA_DIR / "temporal_snapshots.csv"
_COMPETENCIES_PATH = _DATA_DIR / "competencies.csv"
_COURSES_PATH = _DATA_DIR / "courses.csv"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    """Load rows from a CSV file."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_prerequisites() -> List[Dict[str, str]]:
    """Load prerequisite edges from CSV."""
    return _load_csv(_PREREQ_PATH)


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


@router.get("/network")
def get_program_network(
    year: Optional[str] = Query(None, description="Filter by academic year"),
):
    """Cross-program network: nodes = FGOS programs, edges = shared courses."""
    rows = _load_prerequisites()
    if year:
        rows = [r for r in rows if r["year"] == year]

    # Map each course to the set of programs it belongs to
    course_programs: Dict[str, set] = defaultdict(set)
    program_courses: Dict[str, set] = defaultdict(set)
    program_years: Dict[str, set] = defaultdict(set)

    for r in _load_prerequisites():  # unfiltered for year info
        program_years[r["fgos_code"]].add(r["year"])

    for r in rows:
        fgos = r["fgos_code"]
        for name in (r["course_name"], r["prerequisite_name"]):
            course_programs[name].add(fgos)
            program_courses[fgos].add(name)

    # Build program nodes
    cyto_nodes = []
    for fgos, courses in sorted(program_courses.items()):
        area_prefix = fgos.split(".")[0] if "." in fgos else fgos
        cyto_nodes.append({
            "data": {
                "id": fgos,
                "label": fgos,
                "course_count": len(courses),
                "area_prefix": area_prefix,
                "years": sorted(program_years.get(fgos, set())),
            }
        })

    # Build edges: shared courses between program pairs
    cyto_edges = []
    programs = sorted(program_courses.keys())
    for a, b in combinations(programs, 2):
        shared = program_courses[a] & program_courses[b]
        if shared:
            cyto_edges.append({
                "data": {
                    "source": a,
                    "target": b,
                    "shared_count": len(shared),
                    "shared_courses": sorted(shared)[:10],  # cap for payload size
                }
            })

    return {
        "elements": {"nodes": cyto_nodes, "edges": cyto_edges},
        "stats": {
            "programs": len(cyto_nodes),
            "edges": len(cyto_edges),
            "year": year,
        },
    }


@router.get("/temporal-diff")
def get_temporal_diff(
    fgos_code: str = Query(..., description="FGOS program code"),
    year_a: str = Query("2024", description="Base year"),
    year_b: str = Query("2025", description="Comparison year"),
):
    """Prerequisite DAG with temporal change overlay.

    Merges prerequisite graphs from two years and annotates each node
    with its change_type (new, modified, stable, removed).
    """
    all_rows = _load_prerequisites()
    snapshots = _load_csv(_TEMPORAL_PATH)

    rows_a = [r for r in all_rows if r["fgos_code"] == fgos_code and r["year"] == year_a]
    rows_b = [r for r in all_rows if r["fgos_code"] == fgos_code and r["year"] == year_b]

    # Build snapshot lookup: (fgos_code, course_name, year) -> change_type
    snap_lookup: Dict[tuple, str] = {}
    for s in snapshots:
        key = (s.get("fgos_code", ""), s.get("course_name", ""), s.get("year", ""))
        snap_lookup[key] = s.get("change_type", "unknown")

    # Collect unique courses per year
    def _courses_from_rows(rows: list) -> set:
        names: set = set()
        for r in rows:
            names.add(r["course_name"])
            names.add(r["prerequisite_name"])
        return names

    courses_a = _courses_from_rows(rows_a)
    courses_b = _courses_from_rows(rows_b)
    all_courses = courses_a | courses_b

    # Determine change_type for each course
    def _change_type(name: str) -> str:
        # First try snapshot data
        ct = snap_lookup.get((fgos_code, name, year_b))
        if ct:
            return ct
        # Fallback: derive from set membership
        if name in courses_b and name not in courses_a:
            return "new"
        if name in courses_a and name not in courses_b:
            return "removed"
        return "stable"

    # Build union DAG
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: list = []
    in_degree: Dict[str, int] = defaultdict(int)
    out_degree: Dict[str, int] = defaultdict(int)
    edge_set: set = set()

    for r in rows_a + rows_b:
        if r["fgos_code"] != fgos_code:
            continue
        src, tgt = r["prerequisite_name"], r["course_name"]
        for name in (src, tgt):
            if name not in nodes:
                nodes[name] = {"id": name, "label": name, "change_type": _change_type(name)}
        edge_key = (src, tgt)
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            year_tag = r["year"]
            in_both = any(
                x["prerequisite_name"] == src and x["course_name"] == tgt
                for x in (rows_a if r in rows_b else rows_b)
            )
            edges.append({"source": src, "target": tgt, "year": year_tag, "in_both": in_both})
            in_degree[tgt] += 1
            out_degree[src] += 1

    # Also add courses that appear in only one year but not as edges
    for name in all_courses:
        if name not in nodes:
            nodes[name] = {"id": name, "label": name, "change_type": _change_type(name)}

    cyto_nodes = []
    for nid, ndata in nodes.items():
        cyto_nodes.append({
            "data": {
                "id": nid,
                "label": nid,
                "change_type": ndata["change_type"],
                "in_degree": in_degree.get(nid, 0),
                "out_degree": out_degree.get(nid, 0),
                "is_root": in_degree.get(nid, 0) == 0,
                "is_leaf": out_degree.get(nid, 0) == 0,
            }
        })

    cyto_edges = [{"data": e} for e in edges]

    change_counts = defaultdict(int)
    for n in cyto_nodes:
        change_counts[n["data"]["change_type"]] += 1

    return {
        "elements": {"nodes": cyto_nodes, "edges": cyto_edges},
        "stats": {
            "nodes": len(cyto_nodes),
            "edges": len(cyto_edges),
            "fgos_code": fgos_code,
            "year_a": year_a,
            "year_b": year_b,
            "changes": dict(change_counts),
        },
    }


@router.get("/stats")
def get_graph_stats():
    """Aggregate dataset statistics from all MISIS CSV files."""
    prereqs = _load_prerequisites()
    snapshots = _load_csv(_TEMPORAL_PATH)
    competencies = _load_csv(_COMPETENCIES_PATH)
    courses = _load_csv(_COURSES_PATH)

    # Programs and years
    programs: Dict[str, set] = defaultdict(set)
    all_course_names: set = set()
    for r in prereqs:
        programs[r["fgos_code"]].add(r["year"])
        all_course_names.add(r["course_name"])
        all_course_names.add(r["prerequisite_name"])

    # Courses per year
    courses_per_year: Dict[str, set] = defaultdict(set)
    for r in prereqs:
        courses_per_year[r["year"]].add(r["course_name"])
        courses_per_year[r["year"]].add(r["prerequisite_name"])

    # Programs per year
    programs_per_year: Dict[str, set] = defaultdict(set)
    for r in prereqs:
        programs_per_year[r["year"]].add(r["fgos_code"])

    # Shared courses (appear in 2+ programs)
    course_to_programs: Dict[str, set] = defaultdict(set)
    for r in prereqs:
        for name in (r["course_name"], r["prerequisite_name"]):
            course_to_programs[name].add(r["fgos_code"])
    shared_count = sum(1 for progs in course_to_programs.values() if len(progs) >= 2)

    # Competency distribution
    comp_dist: Dict[str, int] = defaultdict(int)
    for c in competencies:
        ctype = c.get("competency_type", "unknown")
        comp_dist[ctype] += 1

    # Temporal change counts
    change_dist: Dict[str, int] = defaultdict(int)
    for s in snapshots:
        change_dist[s.get("change_type", "unknown")] += 1

    return {
        "total_courses": len(all_course_names),
        "total_courses_catalog": len(courses),
        "total_programs": len(programs),
        "total_prereq_edges": len(prereqs),
        "total_competencies": len(competencies),
        "total_snapshots": len(snapshots),
        "shared_across_programs": shared_count,
        "competency_distribution": dict(comp_dist),
        "temporal_changes": dict(change_dist),
        "courses_per_year": {y: len(cs) for y, cs in sorted(courses_per_year.items())},
        "programs_per_year": {y: len(ps) for y, ps in sorted(programs_per_year.items())},
        "years": sorted({r["year"] for r in prereqs}),
    }
