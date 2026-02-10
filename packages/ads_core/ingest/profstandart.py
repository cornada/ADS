"""Russian Professional Standards (Профстандарты) ingestion.

Parses the Ministry of Labor professional standards registry into
structured CSV and ADS artifacts. These are the Russian equivalent
of O*NET — government-defined occupational standards with competency
requirements mapped to FGOS educational standards.

Source: https://profstandart.rosmintrud.ru/
Registry download: https://classinform.ru/profstandarty/reestr_professionalnyh_standartov.xls

Usage:
    python -m ads_core.ingest.profstandart
    python -m ads_core.ingest.profstandart --xls /path/to/registry.xls
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DIR = _REPO_ROOT / "data" / "raw" / "profstandart"
_REGISTRY_URL = "https://classinform.ru/profstandarty/reestr_professionalnyh_standartov.xls"

# Professional standard code format: NN.NNN (e.g. 06.001)
_PS_CODE_RE = re.compile(r"^\d{2}\.\d{3}$")

# Area code prefix → FGOS direction mapping (approximate)
# PS area codes map to broad FGOS direction prefixes
_AREA_TO_FGOS_PREFIX = {
    "Образование и наука": ["44", "01"],
    "Здравоохранение": ["31", "32", "33"],
    "Связь, информационные и коммуникационные технологии": ["09", "10", "11"],
    "Металлургическое производство": ["22"],
    "Атомная промышленность": ["14", "16"],
    "Электроэнергетика": ["13"],
    "Производство машин и оборудования": ["15"],
    "Строительство и жилищно-коммунальное хозяйство": ["08"],
    "Транспорт": ["23", "26"],
    "Сквозные виды профессиональной деятельности": ["38", "40"],
}

FIELDNAMES = [
    "ps_code", "name", "area", "activity_type",
    "reg_number", "order_date", "responsible_org",
]


def _short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def download_registry(output_path: Path) -> Path:
    """Download the professional standards registry XLS."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading registry from %s", _REGISTRY_URL)
    with urlopen(_REGISTRY_URL) as resp:
        output_path.write_bytes(resp.read())
    logger.info("Saved: %s", output_path)
    return output_path


def parse_registry(xls_path: Path) -> List[Dict[str, Any]]:
    """Parse the professional standards XLS into structured records.

    Args:
        xls_path: Path to the registry XLS file.

    Returns:
        List of professional standard dicts.
    """
    import pandas as pd

    df = pd.read_excel(xls_path, header=None, skiprows=5)

    # Column mapping from the XLS structure
    col_map = {
        0: "row_num",
        1: "reg_number",
        2: "ps_code",
        3: "area",
        4: "activity_type",
        5: "name",
        6: "order_number",
        7: "order_date",
        18: "responsible_org",
    }

    df = df.rename(columns=col_map)
    df = df[list(col_map.values())]

    # Filter to valid standard codes
    df = df.dropna(subset=["ps_code"])
    df["ps_code"] = df["ps_code"].astype(str).str.strip()
    df = df[df["ps_code"].str.match(r"\d{2}\.\d{3}", na=False)]

    standards = []
    for _, row in df.iterrows():
        std = {
            "ps_code": row["ps_code"],
            "name": _norm(row.get("name", "")),
            "area": _norm(row.get("area", "")),
            "activity_type": _norm(row.get("activity_type", ""))[:200],
            "reg_number": str(row.get("reg_number", "")).strip(),
            "order_date": str(row.get("order_date", "")).strip()[:10],
            "responsible_org": _norm(row.get("responsible_org", "")),
        }
        if std["name"]:
            standards.append(std)

    logger.info("Parsed %d professional standards", len(standards))
    return standards


def save_csv(standards: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save standards to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(standards)

    logger.info("Wrote %d standards to %s", len(standards), output_path)
    return output_path


def generate_artifacts(
    standards: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate ADS artifacts from professional standards.

    Each standard becomes a JOB_ROLE artifact (parallel to O*NET occupations),
    enabling cross-national comparison of labor market definitions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for std in standards:
            code = std["ps_code"]
            name = std["name"]
            area = std["area"]
            activity = std["activity_type"]

            parts = [
                f"{name}.",
                f"Area: {area}." if area else "",
                f"Activity: {activity}." if activity else "",
                f"Professional standard code: {code}.",
                "Source: Russian Ministry of Labor (Минтруд России).",
            ]
            text = _norm(" ".join(p for p in parts if p))

            aid = f"profstandart:{code}"
            src_url = "https://profstandart.rosmintrud.ru/obshchiy-informatsionnyy-blok/natsionalnyy-reestr-professionalnykh-standartov/reestr-professionalnykh-standartov/"

            prov = {
                "source_url": src_url,
                "fetched_at": now,
                "raw_text_hash": _sha256_text(text),
                "dataset": "profstandart",
                "ps_code": code,
            }

            rec = {
                "artifact_id": aid,
                "type": "JOB_ROLE",
                "source_url": src_url,
                "source_hash": prov["raw_text_hash"],
                "timestamp": now,
                "text": text,
                "metadata": {
                    "institution": "MARKET_RU",
                    "ps_code": code,
                    "name": name,
                    "area": area,
                    "activity_type": activity,
                    "responsible_org": std.get("responsible_org", ""),
                    "country": "RU",
                    "provenance": prov,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote %d artifacts to %s", count, output_path)
    return output_path


def run_ingestion(
    output_dir: Optional[Path] = None,
    *,
    xls_path: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """Run the professional standards ingestion pipeline.

    Args:
        output_dir: Where to write output. Defaults to data/raw/profstandart/.
        xls_path: Path to registry XLS. Will download if not provided.
        verbose: Print progress.

    Returns:
        Path to output CSV.
    """
    if output_dir is None:
        output_dir = _DEFAULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Russian Professional Standards Ingestion")

    # Download if needed
    if xls_path is None:
        xls_path = output_dir / "reestr_profstandartov.xls"
        if not xls_path.exists():
            if verbose:
                print(f"  Downloading registry from {_REGISTRY_URL}...")
            download_registry(xls_path)
        else:
            if verbose:
                print(f"  Using cached: {xls_path}")

    # Parse
    standards = parse_registry(xls_path)

    if verbose:
        areas = set(s["area"] for s in standards)
        print(f"  Standards: {len(standards)}")
        print(f"  Areas: {len(areas)}")

    # Save CSV
    csv_path = save_csv(standards, output_dir / "profstandart.csv")

    # Generate artifacts
    artifacts_path = generate_artifacts(standards, output_dir / "artifacts.jsonl")

    # Manifest
    manifest = {
        "source": "Russian Ministry of Labor Professional Standards",
        "source_url": _REGISTRY_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "standard_count": len(standards),
        "content_hash": hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"  CSV: {csv_path}")
        print(f"  Artifacts: {artifacts_path}")
        print(f"  Manifest: {manifest_path}")
        print(f"  [OK] Profstandart ingestion complete: {len(standards)} standards")

    return csv_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest Russian Professional Standards")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--xls", type=Path, default=None)
    args = parser.parse_args()

    run_ingestion(output_dir=args.output_dir, xls_path=args.xls)
