"""FGOS compliance API endpoints.

Provides FGOS direction → competency coverage analysis:
- Coverage matrix (heatmap data)
- Per-FGOS drill-down reports
- Gap analysis across all directions
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data" / "raw" / "misis"
_COMPETENCIES_PATH = _DATA_DIR / "competencies.csv"
_COURSES_PATH = _DATA_DIR / "courses.csv"
_ARTIFACTS_PATH = _DATA_DIR / "artifacts.jsonl"
_COMP_ARTIFACTS_PATH = _DATA_DIR / "competency_artifacts.jsonl"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


# ── Cached data ──
_cache: Dict[str, Any] = {}


def _get_data():
    """Load and cache all compliance-relevant data."""
    if _cache:
        return _cache

    competencies = _load_csv(_COMPETENCIES_PATH)
    courses = _load_csv(_COURSES_PATH)
    artifacts = _load_jsonl(_ARTIFACTS_PATH)
    comp_artifacts = _load_jsonl(_COMP_ARTIFACTS_PATH)

    # Build FGOS direction → set of competency codes from competencies.csv
    # Each row: course_name, competency_code, competency_type, competency_text, fgos_code, year
    fgos_competencies: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in competencies:
        fgos = row.get("fgos_code", "")
        code = row.get("competency_code", "")
        ctype = row.get("competency_type", "")
        if not fgos or not code:
            continue
        if code not in fgos_competencies[fgos]:
            fgos_competencies[fgos][code] = {
                "code": code,
                "type": ctype,
                "courses": set(),
            }
        course_name = row.get("course_name", "")
        if course_name:
            fgos_competencies[fgos][code]["courses"].add(course_name)

    # Build competency artifact lookup: (code, fgos) → description
    comp_desc: Dict[str, str] = {}
    for art in comp_artifacts:
        meta = art.get("metadata", {})
        code = meta.get("competency_code", "")
        fgos = meta.get("fgos_code", "")
        text = art.get("text", "")
        if code and fgos:
            comp_desc[f"{code}:{fgos}"] = text

    # Build course artifact lookup: artifact_id → metadata
    course_meta: Dict[str, Dict[str, Any]] = {}
    for art in artifacts:
        if art.get("type") == "COURSE":
            aid = art.get("artifact_id", "")
            course_meta[aid] = art.get("metadata", {})

    # Build FGOS names from course departments
    fgos_names: Dict[str, str] = {}
    for row in courses:
        cid = row.get("course_id", "")
        fgos = cid.split(":")[0] if ":" in cid else ""
        dept = row.get("department", "")
        if fgos and dept and fgos not in fgos_names:
            fgos_names[fgos] = dept

    _cache["fgos_competencies"] = fgos_competencies
    _cache["comp_desc"] = comp_desc
    _cache["course_meta"] = course_meta
    _cache["fgos_names"] = fgos_names
    _cache["courses_csv"] = courses
    return _cache


@router.get("/matrix")
def compliance_matrix():
    """Full FGOS → competency coverage matrix for heatmap visualization."""
    data = _get_data()
    fgos_comp = data["fgos_competencies"]
    fgos_names = data["fgos_names"]

    directions = []
    total_comps = 0
    full_coverage = 0
    gap_count = 0

    for fgos in sorted(fgos_comp.keys()):
        comps = fgos_comp[fgos]
        comp_list = []
        covered = 0
        for code, info in sorted(comps.items()):
            count = len(info["courses"])
            status = "covered" if count > 0 else "gap"
            if count > 0:
                covered += 1
            else:
                gap_count += 1
            comp_list.append({
                "code": code,
                "type": info["type"],
                "count": count,
                "status": status,
            })

        total = len(comp_list)
        total_comps += total
        pct = round(covered / total * 100, 1) if total else 0
        if pct == 100:
            full_coverage += 1

        directions.append({
            "fgos_code": fgos,
            "name": fgos_names.get(fgos, fgos),
            "competencies": comp_list,
            "coverage_pct": pct,
            "total": total,
            "covered": covered,
        })

    return {
        "directions": directions,
        "summary": {
            "total_fgos": len(directions),
            "full_coverage": full_coverage,
            "partial_coverage": len(directions) - full_coverage,
            "total_competencies": total_comps,
            "total_gaps": gap_count,
        },
    }


@router.get("/report/{fgos_code}")
def compliance_report(fgos_code: str):
    """Drill-down compliance report for one FGOS direction."""
    data = _get_data()
    fgos_comp = data["fgos_competencies"]
    comp_desc = data["comp_desc"]
    fgos_names = data["fgos_names"]

    if fgos_code not in fgos_comp:
        return {"fgos_code": fgos_code, "programs": [], "competencies": [], "coverage_pct": 0}

    comps = fgos_comp[fgos_code]

    # All courses in this FGOS
    all_courses: Dict[str, List[str]] = defaultdict(list)
    competency_details = []
    covered = 0

    for code, info in sorted(comps.items()):
        course_list = sorted(info["courses"])
        for c in course_list:
            all_courses[c].append(code)

        desc_key = f"{code}:{fgos_code}"
        description = comp_desc.get(desc_key, "")
        status = "covered" if len(course_list) > 0 else "gap"
        if len(course_list) > 0:
            covered += 1

        competency_details.append({
            "code": code,
            "type": info["type"],
            "description": description[:200] if description else "",
            "covering_programs": course_list,
            "count": len(course_list),
            "status": status,
        })

    programs = [
        {"course_name": name, "competency_codes": codes}
        for name, codes in sorted(all_courses.items())
    ]

    total = len(competency_details)
    pct = round(covered / total * 100, 1) if total else 0

    return {
        "fgos_code": fgos_code,
        "fgos_name": fgos_names.get(fgos_code, fgos_code),
        "programs": programs,
        "competencies": competency_details,
        "coverage_pct": pct,
        "summary": {"total": total, "covered": covered, "gaps": total - covered},
    }


@router.get("/gaps")
def compliance_gaps():
    """Find competencies with low coverage across all FGOS directions."""
    data = _get_data()
    fgos_comp = data["fgos_competencies"]

    gaps = []
    by_type: Dict[str, int] = defaultdict(int)

    for fgos in sorted(fgos_comp.keys()):
        comps = fgos_comp[fgos]
        for code, info in sorted(comps.items()):
            count = len(info["courses"])
            if count <= 1:  # gaps = 0 or 1 covering course
                status = "uncovered" if count == 0 else "weak"
                gaps.append({
                    "fgos_code": fgos,
                    "competency_code": code,
                    "type": info["type"],
                    "covering_count": count,
                    "status": status,
                })
                by_type[info["type"]] += 1

    return {
        "gaps": gaps,
        "summary": {
            "total_gaps": len(gaps),
            "by_type": dict(by_type),
        },
    }
