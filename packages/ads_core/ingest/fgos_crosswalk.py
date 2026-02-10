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


# ============================================================================
# Two-hop crosswalk: FGOS → ISCED → CIP → SOC → O*NET
# ============================================================================

# FGOS area prefix → ISCED-F 2013 broad field code
# Based on UNESCO ISCED-F classification and Russian FGOS structure
FGOS_TO_ISCED = {
    "01": "05",  # Math → Natural sciences, mathematics and statistics
    "03": "05",  # Physics → Natural sciences
    "09": "06",  # CS/IT → Information and Communication Technologies
    "10": "06",  # Information security → ICT
    "11": "07",  # Electronics → Engineering, manufacturing
    "12": "07",  # Photonics → Engineering
    "13": "07",  # Electrical engineering → Engineering
    "15": "07",  # Mechanical engineering → Engineering
    "18": "07",  # Chemical technology → Engineering
    "20": "05",  # Ecology → Natural sciences
    "21": "07",  # Mining → Engineering
    "22": "07",  # Metallurgy → Engineering
    "27": "07",  # Quality management → Engineering
    "28": "07",  # Nanotechnology → Engineering
    "29": "07",  # Nuclear engineering → Engineering
    "38": "04",  # Economics → Business, administration
    "45": "02",  # Linguistics → Arts and humanities
}

# ISCED broad field → CIP 2-digit series (NCES classification)
# Based on NCES ISCED↔CIP crosswalk
ISCED_TO_CIP = {
    "02": ["16", "23", "24", "38", "39"],  # Arts & humanities → Languages, English, Liberal arts, Philosophy, Theology
    "03": ["22", "42", "44", "45"],  # Social sciences → Law, Psychology, Public admin, Social sciences
    "04": ["52", "30"],  # Business → Business/management, Multi/interdisciplinary
    "05": ["26", "27", "40", "41"],  # Natural sciences → Biology, Mathematics, Physical sciences, Science tech
    "06": ["11"],  # ICT → Computer and Information Sciences
    "07": ["14", "15"],  # Engineering → Engineering, Engineering technologies
}

# CIP 2-digit → SOC major group (BLS crosswalk)
# Based on BLS CIP-SOC crosswalk 2020
CIP_TO_SOC = {
    "11": ["15-0000"],  # Computer science → Computer and Mathematical
    "14": ["17-0000"],  # Engineering → Architecture and Engineering
    "15": ["17-0000"],  # Engineering tech → Architecture and Engineering
    "16": ["25-0000", "27-0000"],  # Languages → Education, Arts/Media
    "22": ["23-0000"],  # Law → Legal
    "23": ["25-0000", "27-0000"],  # English → Education, Arts
    "26": ["19-0000"],  # Biology → Life/Physical/Social Science
    "27": ["15-0000"],  # Mathematics → Computer and Mathematical
    "30": ["19-0000", "25-0000"],  # Multi/interdisciplinary → Science, Education
    "38": ["25-0000"],  # Philosophy → Education
    "40": ["19-0000"],  # Physical sciences → Life/Physical/Social Science
    "41": ["19-0000"],  # Science technologies → Life/Physical/Social Science
    "42": ["19-0000", "21-0000"],  # Psychology → Science, Community/Social
    "44": ["21-0000"],  # Public admin → Community and Social Service
    "45": ["19-0000"],  # Social sciences → Life/Physical/Social Science
    "52": ["11-0000", "13-0000"],  # Business → Management, Business/Financial
}

# SOC major group → O*NET occupation family prefix
SOC_TO_ONET_PREFIX = {
    "11-0000": "11-",  # Management
    "13-0000": "13-",  # Business and Financial Operations
    "15-0000": "15-",  # Computer and Mathematical
    "17-0000": "17-",  # Architecture and Engineering
    "19-0000": "19-",  # Life, Physical, and Social Science
    "21-0000": "21-",  # Community and Social Service
    "23-0000": "23-",  # Legal
    "25-0000": "25-",  # Educational Instruction and Library
    "27-0000": "27-",  # Arts, Design, Entertainment, Sports, and Media
}


def fgos_to_international(fgos_code: str) -> Dict[str, Any]:
    """Map a single FGOS code through the full crosswalk chain.

    Returns dict with all intermediate codes:
    {fgos, isced, cip_codes, soc_codes, onet_prefixes}
    """
    prefix = fgos_code.split(".")[0]
    level_code = fgos_code.split(".")[1] if "." in fgos_code else "03"
    level = FGOS_LEVELS.get(level_code, f"L{level_code}")

    result = {
        "fgos_code": fgos_code,
        "fgos_level": level,
        "fgos_area_prefix": prefix,
        "isced_code": None,
        "cip_codes": [],
        "soc_codes": [],
        "onet_prefixes": [],
    }

    # Hop 1: FGOS → ISCED
    isced = FGOS_TO_ISCED.get(prefix)
    if not isced:
        return result
    result["isced_code"] = isced

    # Hop 2: ISCED → CIP
    cip_codes = ISCED_TO_CIP.get(isced, [])
    result["cip_codes"] = cip_codes

    # Hop 3: CIP → SOC
    soc_codes = set()
    for cip in cip_codes:
        for soc in CIP_TO_SOC.get(cip, []):
            soc_codes.add(soc)
    result["soc_codes"] = sorted(soc_codes)

    # Hop 4: SOC → O*NET prefix
    onet_prefixes = set()
    for soc in soc_codes:
        pfx = SOC_TO_ONET_PREFIX.get(soc)
        if pfx:
            onet_prefixes.add(pfx)
    result["onet_prefixes"] = sorted(onet_prefixes)

    return result


def build_international_crosswalk(
    fgos_codes: List[str],
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Build full FGOS→ISCED→CIP→SOC→O*NET crosswalk for a list of FGOS codes."""
    rows = []
    for fgos in sorted(set(fgos_codes)):
        mapping = fgos_to_international(fgos)
        rows.append(mapping)

    if verbose:
        mapped = sum(1 for r in rows if r["isced_code"])
        with_soc = sum(1 for r in rows if r["soc_codes"])
        logger.info("International crosswalk: %d FGOS codes → %d with ISCED → %d with SOC",
                     len(rows), mapped, with_soc)
    return rows


def save_international_crosswalk(
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Save international crosswalk to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (full detail)
    json_path = output_dir / "fgos_international_crosswalk.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV (flat, one row per FGOS→SOC pair)
    csv_path = output_dir / "fgos_to_soc.csv"
    fields = ["fgos_code", "fgos_level", "isced_code", "cip_code", "soc_code", "onet_prefix"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["soc_codes"]:
                writer.writerow({
                    "fgos_code": row["fgos_code"], "fgos_level": row["fgos_level"],
                    "isced_code": row["isced_code"] or "", "cip_code": "", "soc_code": "", "onet_prefix": "",
                })
            for soc in row["soc_codes"]:
                onet = SOC_TO_ONET_PREFIX.get(soc, "")
                for cip in row["cip_codes"]:
                    if soc in CIP_TO_SOC.get(cip, []):
                        writer.writerow({
                            "fgos_code": row["fgos_code"], "fgos_level": row["fgos_level"],
                            "isced_code": row["isced_code"], "cip_code": cip,
                            "soc_code": soc, "onet_prefix": onet,
                        })

    logger.info("Saved: %s, %s", json_path, csv_path)
    return json_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_crosswalk()

    # Also build international crosswalk if MISIS data exists
    courses_path = _REPO_ROOT / "data" / "raw" / "misis" / "courses.csv"
    if courses_path.exists():
        import re
        fgos_codes = set()
        with courses_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = f"{row.get('description', '')} {row.get('department', '')}"
                codes = re.findall(r"\d{2}\.\d{2}\.\d{2}", text)
                fgos_codes.update(codes)

        if fgos_codes:
            out_dir = _REPO_ROOT / "data" / "processed" / "crosswalks"
            rows = build_international_crosswalk(list(fgos_codes), verbose=True)
            save_international_crosswalk(rows, out_dir)
            print(f"\nInternational crosswalk: {len(rows)} FGOS codes mapped")
    else:
        # Demo with known MISIS FGOS codes
        demo_codes = ["09.04.01", "09.03.01", "22.04.01", "01.03.04", "38.04.02"]
        print("\nDemo international crosswalk:")
        for code in demo_codes:
            m = fgos_to_international(code)
            print(f"  {code} → ISCED:{m['isced_code']} → CIP:{m['cip_codes']} → SOC:{m['soc_codes']}")
