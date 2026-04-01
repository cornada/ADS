"""Strategy & program detail API endpoints.

Serves pre-computed Pareto analysis from experiment results,
enriched with course metadata for strategic insights.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _REPO_ROOT / "experiments" / "reports" / "20260219_033212"
_ARTIFACTS_PATH = _REPO_ROOT / "data" / "raw" / "misis" / "artifacts.jsonl"

_OBJ_KEYS = ["market", "mission", "university", "competency"]

# ── Cached data ──
_cache: Dict[str, Any] = {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_data():
    """Load and cache experiment results + artifact metadata."""
    if _cache:
        return _cache

    # Load experiment results
    results_path = _RESULTS_DIR / "results.json"
    pareto_path = _RESULTS_DIR / "pareto.json"

    results_data = {}
    pareto_data = {}
    if results_path.exists():
        with results_path.open() as f:
            results_data = json.load(f)
    if pareto_path.exists():
        with pareto_path.open() as f:
            pareto_data = json.load(f)

    # Build artifact metadata lookup
    artifacts = _load_jsonl(_ARTIFACTS_PATH)
    meta_lookup: Dict[str, Dict[str, Any]] = {}
    for art in artifacts:
        if art.get("type") == "COURSE":
            aid = art.get("artifact_id", "")
            m = art.get("metadata", {})
            meta_lookup[aid] = {
                "name": m.get("course_name", aid),
                "department": m.get("department", ""),
                "fgos_code": m.get("fgos_code", ""),
                "competency_codes": m.get("competency_codes", ""),
                "level": m.get("level", ""),
                "year": m.get("year", ""),
            }

    # Build pareto set
    pareto_ids = set()
    for p in pareto_data.get("pareto", []):
        pareto_ids.add(p["option_id"])

    # Build enriched options list
    options = []
    for r in results_data.get("results", []):
        oid = r["option_id"]
        meta = meta_lookup.get(oid, {})
        options.append({
            "option_id": oid,
            "name": meta.get("name", oid),
            "department": meta.get("department", ""),
            "fgos_code": meta.get("fgos_code", ""),
            "competency_codes": meta.get("competency_codes", ""),
            "level": meta.get("level", ""),
            "year": meta.get("year", ""),
            "objectives": r.get("objectives", {}),
            "is_pareto": oid in pareto_ids,
            "feasible": r.get("feasible", True),
        })

    _cache["options"] = options
    _cache["pareto_ids"] = pareto_ids
    _cache["meta_lookup"] = meta_lookup
    _cache["obj_keys"] = pareto_data.get("keys", _OBJ_KEYS)
    return _cache


def _obj_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Euclidean distance in objective space."""
    total = 0.0
    for k in _OBJ_KEYS:
        total += (a.get(k, 0) - b.get(k, 0)) ** 2
    return math.sqrt(total)


@router.get("/pareto")
def strategy_pareto():
    """Full Pareto analysis with enriched option details."""
    data = _get_data()
    options = data["options"]
    pareto_ids = data["pareto_ids"]

    pareto_count = len(pareto_ids)
    feasible_count = sum(1 for o in options if o["feasible"])

    # Objective stats
    obj_stats: Dict[str, Dict[str, float]] = {}
    for key in _OBJ_KEYS:
        vals = [o["objectives"].get(key, 0) for o in options if o["objectives"]]
        if vals:
            obj_stats[key] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
            }

    return {
        "options": options,
        "pareto_ids": sorted(pareto_ids),
        "objective_keys": _OBJ_KEYS,
        "stats": {
            "total": len(options),
            "pareto_count": pareto_count,
            "pareto_ratio": round(pareto_count / len(options), 4) if options else 0,
            "feasible_count": feasible_count,
            "objective_stats": obj_stats,
        },
    }


@router.get("/insights")
def strategy_insights():
    """Pre-computed strategic insights: top/bottom programs, trade-offs."""
    data = _get_data()
    options = data["options"]

    def _top_n(key: str, n: int = 10):
        ranked = sorted(options, key=lambda o: o["objectives"].get(key, 0), reverse=True)
        return [
            {"option_id": o["option_id"], "name": o["name"], "fgos_code": o["fgos_code"],
             "score": round(o["objectives"].get(key, 0), 4), "is_pareto": o["is_pareto"]}
            for o in ranked[:n]
        ]

    # Bottom programs: lowest average across all objectives
    def _avg_score(o):
        vals = [o["objectives"].get(k, 0) for k in _OBJ_KEYS]
        return sum(vals) / len(vals) if vals else 0

    bottom = sorted(options, key=_avg_score)[:10]
    bottom_list = [
        {"option_id": o["option_id"], "name": o["name"], "fgos_code": o["fgos_code"],
         "avg_score": round(_avg_score(o), 4), "objectives": {k: round(v, 4) for k, v in o["objectives"].items()}}
        for o in bottom
    ]

    # Trade-offs: programs where max objective - min objective is largest
    def _spread(o):
        vals = [o["objectives"].get(k, 0) for k in _OBJ_KEYS]
        return max(vals) - min(vals) if vals else 0

    tradeoffs = sorted(options, key=_spread, reverse=True)[:10]
    tradeoff_list = [
        {"option_id": o["option_id"], "name": o["name"], "fgos_code": o["fgos_code"],
         "spread": round(_spread(o), 4),
         "strongest": max(_OBJ_KEYS, key=lambda k: o["objectives"].get(k, 0)),
         "weakest": min(_OBJ_KEYS, key=lambda k: o["objectives"].get(k, 0)),
         "objectives": {k: round(v, 4) for k, v in o["objectives"].items()}}
        for o in tradeoffs
    ]

    return {
        "top_market": _top_n("market"),
        "top_mission": _top_n("mission"),
        "top_university": _top_n("university"),
        "top_competency": _top_n("competency"),
        "bottom": bottom_list,
        "tradeoffs": tradeoff_list,
    }


@router.get("/dominated")
def strategy_dominated():
    """Dominated programs with nearest Pareto neighbor and improvement gap."""
    data = _get_data()
    options = data["options"]
    pareto_ids = data["pareto_ids"]

    pareto_options = [o for o in options if o["is_pareto"]]
    dominated = []

    for o in options:
        if o["is_pareto"]:
            continue
        # Find nearest Pareto neighbor
        best_dist = float("inf")
        best_neighbor = None
        for p in pareto_options:
            d = _obj_distance(o["objectives"], p["objectives"])
            if d < best_dist:
                best_dist = d
                best_neighbor = p

        gap = {}
        if best_neighbor:
            for k in _OBJ_KEYS:
                gap[k] = round(best_neighbor["objectives"].get(k, 0) - o["objectives"].get(k, 0), 4)

        dominated.append({
            "option_id": o["option_id"],
            "name": o["name"],
            "fgos_code": o["fgos_code"],
            "objectives": {k: round(v, 4) for k, v in o["objectives"].items()},
            "nearest_pareto": best_neighbor["option_id"] if best_neighbor else None,
            "nearest_pareto_name": best_neighbor["name"] if best_neighbor else None,
            "distance": round(best_dist, 4),
            "gap": gap,
        })

    # Sort by distance (biggest improvement potential first)
    dominated.sort(key=lambda x: -x["distance"])

    return {
        "dominated": dominated,
        "stats": {
            "total_dominated": len(dominated),
            "avg_distance": round(sum(d["distance"] for d in dominated) / len(dominated), 4) if dominated else 0,
        },
    }


@router.get("/program/{option_id:path}")
def strategy_program(option_id: str):
    """Single program detail card."""
    data = _get_data()
    options = data["options"]
    pareto_ids = data["pareto_ids"]

    option = None
    for o in options:
        if o["option_id"] == option_id:
            option = o
            break

    if not option:
        return {"error": "Program not found", "option_id": option_id}

    # Find neighbors (closest in objective space)
    pareto_options = [o for o in options if o["is_pareto"] and o["option_id"] != option_id]
    neighbors = []
    for p in pareto_options:
        d = _obj_distance(option["objectives"], p["objectives"])
        neighbors.append({"option_id": p["option_id"], "name": p["name"], "distance": round(d, 4)})
    neighbors.sort(key=lambda x: x["distance"])

    # Radar data
    radar = {k: round(option["objectives"].get(k, 0), 4) for k in _OBJ_KEYS}

    # Rank among all options per objective
    ranks = {}
    for k in _OBJ_KEYS:
        sorted_by_k = sorted(options, key=lambda o: o["objectives"].get(k, 0), reverse=True)
        for i, o in enumerate(sorted_by_k):
            if o["option_id"] == option_id:
                ranks[k] = i + 1
                break

    return {
        **option,
        "radar": radar,
        "ranks": ranks,
        "total_programs": len(options),
        "nearest_neighbors": neighbors[:5],
    }


@router.get("/programs")
def strategy_programs(
    search: Optional[str] = Query(None, description="Search by name"),
    fgos_code: Optional[str] = Query(None, description="Filter by FGOS code"),
    pareto_only: bool = Query(False, description="Only Pareto-optimal programs"),
    sort_by: str = Query("market", description="Sort by objective"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Searchable, filterable program list."""
    data = _get_data()
    options = data["options"]

    filtered = options

    if search:
        search_lower = search.lower()
        filtered = [o for o in filtered if search_lower in o["name"].lower()]

    if fgos_code:
        filtered = [o for o in filtered if o["fgos_code"] == fgos_code]

    if pareto_only:
        filtered = [o for o in filtered if o["is_pareto"]]

    # Sort
    if sort_by in _OBJ_KEYS:
        filtered.sort(key=lambda o: o["objectives"].get(sort_by, 0), reverse=True)
    elif sort_by == "name":
        filtered.sort(key=lambda o: o["name"])

    total = len(filtered)
    page = filtered[offset:offset + limit]

    # Summarize with rounded objectives
    results = []
    for o in page:
        results.append({
            "option_id": o["option_id"],
            "name": o["name"],
            "department": o["department"],
            "fgos_code": o["fgos_code"],
            "objectives": {k: round(v, 4) for k, v in o["objectives"].items()},
            "is_pareto": o["is_pareto"],
        })

    # Unique FGOS codes for filter dropdown
    all_fgos = sorted(set(o["fgos_code"] for o in options if o["fgos_code"]))

    return {
        "programs": results,
        "total": total,
        "offset": offset,
        "limit": limit,
        "fgos_codes": all_fgos,
    }
