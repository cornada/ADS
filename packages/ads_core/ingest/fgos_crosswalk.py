"""FGOS → Profstandart crosswalk for Russian education-labor market mapping.

Maps FGOS educational direction codes (e.g., 09.04.01 = Informatics & CS)
to Russian Professional Standard areas (e.g., 06 = ICT), enabling
direct comparison between MISIS curricula and labor market requirements.

This is the Russian equivalent of the US CIP→SOC→O*NET crosswalk.

Usage:
    python -m ads_core.ingest.fgos_crosswalk
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# FGOS direction prefix → Profstandart area code(s)
# Based on Минобрнауки ↔ Минтруд correspondence tables
# FGOS prefix = first 2 digits of direction code (e.g., 09 from 09.04.01)
FGOS_TO_PS_AREA = {
    # Математика и информатика
    "01": ["01", "06"],  # Math → Education + ICT
    # Физика
    "03": ["01", "24", "29"],  # Physics → Education + Nuclear + Electronics
    # Информатика и ВТ
    "09": ["06", "40"],  # CS/IT → ICT + Cross-cutting
    # Электроника и наноэлектроника
    "11": ["29", "06"],  # Electronics → Electronics + ICT
    # Экология и природопользование
    "20": ["18", "19", "26"],  # Ecology → Mining + Oil&Gas + Chemical
    # Горное дело
    "21": ["18", "19"],  # Mining → Mining + Oil&Gas
    # Металлургия
    "22": ["27"],  # Metallurgy → Metallurgy
    # Нанотехнологии и наноматериалы
    "28": ["27", "29", "26"],  # Nano → Metallurgy + Electronics + Chemical
    # Управление качеством
    "27": ["40", "28"],  # Quality mgmt → Cross-cutting + Machinery
    # Экономика и менеджмент
    "38": ["08", "07", "40"],  # Economics → Finance + Admin + Cross-cutting
    # Лингвистика
    "45": ["01", "11"],  # Linguistics → Education + Media
}

# FGOS level mapping (middle 2 digits)
FGOS_LEVELS = {
    "03": "bachelor",
    "04": "master",
    "05": "specialist",  # 5-year specialist degree
    "06": "phd",
}

CROSSWALK_FIELDS = [
    "fgos_code", "fgos_level", "fgos_area_prefix",
    "ps_area_code", "ps_area_name",
    "match_type", "confidence",
]


def build_crosswalk(
    courses_path: Path,
    profstandart_path: Path,
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Build FGOS → Profstandart crosswalk from MISIS course data.

    Returns list of mapping rows.
    """
    import re

    # Load profstandart areas
    ps_areas: Dict[str, str] = {}
    with profstandart_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["ps_code"]
            prefix = code.split(".")[0]
            if prefix not in ps_areas:
                ps_areas[prefix] = row["area"]

    # Extract FGOS codes from MISIS
    fgos_codes = set()
    with courses_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = f"{row.get('description', '')} {row.get('department', '')}"
            codes = re.findall(r"\d{2}\.\d{2}\.\d{2}", text)
            fgos_codes.update(codes)

    if verbose:
        print(f"  FGOS codes found: {len(fgos_codes)}")
        print(f"  PS areas: {len(ps_areas)}")

    # Build crosswalk
    rows = []
    for fgos in sorted(fgos_codes):
        parts = fgos.split(".")
        area_prefix = parts[0]
        level_code = parts[1]
        level = FGOS_LEVELS.get(level_code, f"L{level_code}")

        ps_area_codes = FGOS_TO_PS_AREA.get(area_prefix, [])

        if ps_area_codes:
            for ps_code in ps_area_codes:
                ps_name = ps_areas.get(ps_code, "Unknown")
                rows.append({
                    "fgos_code": fgos,
                    "fgos_level": level,
                    "fgos_area_prefix": area_prefix,
                    "ps_area_code": ps_code,
                    "ps_area_name": ps_name,
                    "match_type": "rule_based",
                    "confidence": "high",
                })
        else:
            rows.append({
                "fgos_code": fgos,
                "fgos_level": level,
                "fgos_area_prefix": area_prefix,
                "ps_area_code": "",
                "ps_area_name": "",
                "match_type": "unmapped",
                "confidence": "none",
            })

    return rows


def save_crosswalk(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save crosswalk to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CROSSWALK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def enrich_profstandart_with_fgos(
    crosswalk: List[Dict[str, Any]],
    profstandart_path: Path,
) -> Dict[str, List[str]]:
    """Map each profstandart to its relevant FGOS codes.

    Returns: {ps_code: [fgos_code, ...]}
    """
    ps_to_fgos: Dict[str, List[str]] = {}

    # Build reverse mapping: ps_area → [fgos_codes]
    area_to_fgos: Dict[str, List[str]] = {}
    for row in crosswalk:
        ps_area = row.get("ps_area_code", "")
        fgos = row.get("fgos_code", "")
        if ps_area and fgos:
            area_to_fgos.setdefault(ps_area, []).append(fgos)

    with profstandart_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ps_code = row["ps_code"]
            area_prefix = ps_code.split(".")[0]
            fgos_list = area_to_fgos.get(area_prefix, [])
            if fgos_list:
                ps_to_fgos[ps_code] = sorted(set(fgos_list))

    return ps_to_fgos


def run_crosswalk(
    output_dir: Path | None = None,
    *,
    verbose: bool = True,
) -> Path:
    """Build and save the FGOS→Profstandart crosswalk."""
    courses_path = _REPO_ROOT / "data" / "raw" / "misis" / "courses.csv"
    profstandart_path = _REPO_ROOT / "data" / "raw" / "profstandart" / "profstandart.csv"

    if output_dir is None:
        output_dir = _REPO_ROOT / "data" / "processed" / "crosswalks"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("FGOS → Profstandart Crosswalk")

    crosswalk = build_crosswalk(courses_path, profstandart_path, verbose=verbose)

    # Save crosswalk CSV
    xwalk_path = save_crosswalk(crosswalk, output_dir / "fgos_to_profstandart.csv")

    # Build reverse mapping
    ps_to_fgos = enrich_profstandart_with_fgos(crosswalk, profstandart_path)

    # Save reverse mapping
    reverse_path = output_dir / "profstandart_to_fgos.json"
    reverse_path.write_text(
        json.dumps(ps_to_fgos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if verbose:
        mapped = [r for r in crosswalk if r["match_type"] != "unmapped"]
        unmapped = [r for r in crosswalk if r["match_type"] == "unmapped"]
        print(f"  Mapped: {len(mapped)} FGOS→PS links")
        print(f"  Unmapped: {len(unmapped)} FGOS codes")
        print(f"  PS standards with FGOS links: {len(ps_to_fgos)}")
        print(f"  Crosswalk: {xwalk_path}")
        print(f"  Reverse map: {reverse_path}")
        print("  [OK] Crosswalk complete")

    return xwalk_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_crosswalk()
