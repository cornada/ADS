"""ESCO (European Skills/Competences/Qualifications/Occupations) ingestion.

Fetches occupation profiles and their linked skills from the ESCO REST API
to build a European labor market signal, complementing O*NET (US) and
Profstandart (RU).

API docs: https://ec.europa.eu/esco/api/doc/esco-api-further-doc.html
Public endpoint, no auth required.

Usage:
    python -m ads_core.ingest.esco
    python -m ads_core.ingest.esco --isco-groups 25,21 --max-occupations 200
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DIR = _REPO_ROOT / "data" / "raw" / "esco"
_API_BASE = "https://ec.europa.eu/esco/api"

# ISCO-08 groups relevant to MISIS/ADS universities
# These map to ESCO occupation subtrees
RELEVANT_ISCO_GROUPS = {
    "21": "Science and engineering professionals",
    "25": "Information and communications technology professionals",
    "24": "Business and administration professionals",
    "31": "Science and engineering associate professionals",
    "35": "Information and communications technicians",
    "22": "Health professionals",
    "23": "Teaching professionals",
}

# Focused search queries for MISIS-relevant occupations
SEARCH_QUERIES = [
    "software developer",
    "data scientist",
    "materials engineer",
    "metallurgist",
    "mining engineer",
    "electronics engineer",
    "nanotechnology",
    "energy engineer",
    "environmental engineer",
    "machine learning",
    "DevOps engineer",
    "systems administrator",
    "quality engineer",
    "research scientist",
    "cybersecurity",
]

OCCUPATION_FIELDS = [
    "uri", "title", "isco_group", "isco_code",
    "description", "essential_skills", "optional_skills",
    "essential_skill_count", "optional_skill_count",
]

SKILL_FIELDS = [
    "uri", "title", "skill_type", "reuse_level",
    "occupation_count",
]


def _api_get(endpoint: str, params: Optional[Dict[str, str]] = None) -> Any:
    """Make a GET request to ESCO API."""
    url = f"{_API_BASE}{endpoint}"
    if params:
        url = f"{url}?{urlencode(params)}"

    req = Request(url, headers={
        "User-Agent": "ADS-Research/1.0 (academic research)",
        "Accept": "application/json",
    })
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_occupations(
    query: str,
    *,
    limit: int = 20,
    offset: int = 0,
    language: str = "en",
    delay: float = 0.2,
) -> List[Dict[str, Any]]:
    """Search ESCO occupations by text query.

    Returns list of occupation stubs (uri, title).
    """
    results = []

    try:
        data = _api_get("/search", {
            "text": query,
            "type": "occupation",
            "language": language,
            "limit": str(limit),
            "offset": str(offset),
        })
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("ESCO search failed for '%s': %s", query, e)
        return results

    embedded = data.get("_embedded", {})
    for item in embedded.get("results", []):
        results.append({
            "uri": item.get("uri", ""),
            "title": item.get("title", ""),
        })

    return results


def fetch_occupation(uri: str, *, language: str = "en") -> Optional[Dict[str, Any]]:
    """Fetch full occupation profile including linked skills."""
    try:
        data = _api_get("/resource/occupation", {
            "uri": uri,
            "language": language,
        })
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch occupation %s: %s", uri, e)
        return None

    links = data.get("_links", {})

    # Extract ISCO code
    isco_groups = links.get("broaderIscoGroup", [])
    isco_code = ""
    isco_title = ""
    if isco_groups:
        isco_code = isco_groups[0].get("code", "")
        isco_title = isco_groups[0].get("title", "")

    # Extract essential skills
    essential = []
    for s in links.get("hasEssentialSkill", []):
        essential.append({
            "uri": s.get("uri", ""),
            "title": s.get("title", ""),
            "skill_type": s.get("skillType", ""),
        })

    # Extract optional skills
    optional = []
    for s in links.get("hasOptionalSkill", []):
        optional.append({
            "uri": s.get("uri", ""),
            "title": s.get("title", ""),
            "skill_type": s.get("skillType", ""),
        })

    # Description — may be a dict of language keys or a string
    raw_desc = data.get("description", {})
    if isinstance(raw_desc, dict):
        desc = raw_desc.get(language, "")
    else:
        desc = str(raw_desc) if raw_desc else ""

    return {
        "uri": uri,
        "title": data.get("title", ""),
        "isco_group": isco_title,
        "isco_code": isco_code,
        "description": desc,
        "essential_skills": essential,
        "optional_skills": optional,
        "essential_skill_count": len(essential),
        "optional_skill_count": len(optional),
    }


def save_occupations(occupations: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save occupations to CSV (skills as comma-separated names)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OCCUPATION_FIELDS)
        writer.writeheader()
        for occ in occupations:
            row = {
                "uri": occ["uri"],
                "title": occ["title"],
                "isco_group": occ.get("isco_group", ""),
                "isco_code": occ.get("isco_code", ""),
                "description": str(occ.get("description", ""))[:500],
                "essential_skills": ",".join(
                    s["title"] for s in occ.get("essential_skills", [])
                ),
                "optional_skills": ",".join(
                    s["title"] for s in occ.get("optional_skills", [])
                ),
                "essential_skill_count": occ.get("essential_skill_count", 0),
                "optional_skill_count": occ.get("optional_skill_count", 0),
            }
            writer.writerow(row)

    return output_path


def aggregate_skills(occupations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate skills across occupations."""
    from collections import Counter

    skill_counts: Counter = Counter()
    skill_info: Dict[str, Dict[str, str]] = {}

    for occ in occupations:
        for s in occ.get("essential_skills", []):
            title = s["title"]
            skill_counts[title] += 1
            skill_info[title] = {
                "uri": s.get("uri", ""),
                "skill_type": s.get("skill_type", ""),
                "reuse_level": "essential",
            }
        for s in occ.get("optional_skills", []):
            title = s["title"]
            skill_counts[title] += 1
            if title not in skill_info:
                skill_info[title] = {
                    "uri": s.get("uri", ""),
                    "skill_type": s.get("skill_type", ""),
                    "reuse_level": "optional",
                }

    rows = []
    for title, count in skill_counts.most_common():
        info = skill_info.get(title, {})
        rows.append({
            "uri": info.get("uri", ""),
            "title": title,
            "skill_type": info.get("skill_type", ""),
            "reuse_level": info.get("reuse_level", ""),
            "occupation_count": count,
        })

    return rows


def save_skills(skills: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save aggregated skills to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SKILL_FIELDS)
        writer.writeheader()
        writer.writerows(skills)

    return output_path


def generate_artifacts(occupations: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate ADS artifacts from ESCO occupations.

    Each occupation becomes a JOB_ROLE artifact with EU labor market tagging.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for occ in occupations:
            title = occ["title"]
            raw_desc = occ.get("description", "")
            desc = str(raw_desc) if raw_desc else ""
            isco = occ.get("isco_group", "")
            essential = [s["title"] for s in occ.get("essential_skills", [])]
            optional = [s["title"] for s in occ.get("optional_skills", [])]

            parts = [
                f"{title}.",
                f"ISCO group: {isco}." if isco else "",
                f"Description: {desc[:300]}." if desc else "",
                f"Essential skills: {', '.join(essential[:15])}."
                if essential else "",
                "Source: ESCO (European Commission).",
            ]
            text = " ".join(p for p in parts if p)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            # Extract short ID from URI
            uri = occ["uri"]
            short_id = uri.rsplit("/", 1)[-1][:12] if "/" in uri else uri[:12]
            aid = f"esco:{short_id}"

            rec = {
                "artifact_id": aid,
                "type": "JOB_ROLE",
                "source_url": uri,
                "source_hash": text_hash,
                "timestamp": now,
                "text": text,
                "metadata": {
                    "institution": "MARKET_EU",
                    "title": title,
                    "isco_code": occ.get("isco_code", ""),
                    "isco_group": isco,
                    "essential_skill_count": len(essential),
                    "optional_skill_count": len(optional),
                    "country": "EU",
                    "provenance": {
                        "source_url": uri,
                        "fetched_at": now,
                        "raw_text_hash": text_hash,
                        "dataset": "esco",
                    },
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote %d artifacts to %s", count, output_path)
    return output_path


def run_ingestion(
    output_dir: Optional[Path] = None,
    *,
    queries: Optional[List[str]] = None,
    max_per_query: int = 20,
    delay: float = 0.3,
    verbose: bool = True,
) -> Path:
    """Run ESCO occupation ingestion.

    Args:
        output_dir: Where to write output.
        queries: Search queries. Defaults to SEARCH_QUERIES.
        max_per_query: Max occupations per query.
        delay: API rate limiting delay.
        verbose: Print progress.

    Returns:
        Path to occupations CSV.
    """
    if output_dir is None:
        output_dir = _DEFAULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = SEARCH_QUERIES

    if verbose:
        print("ESCO Occupation Ingestion")
        print(f"  Queries: {len(queries)}")
        print(f"  Max per query: {max_per_query}")

    all_occupations: List[Dict[str, Any]] = []
    seen_uris: set = set()

    for query in queries:
        if verbose:
            print(f"  Searching: '{query}'...", end=" ", flush=True)

        stubs = search_occupations(query, limit=max_per_query, delay=delay)
        new = 0

        for stub in stubs:
            uri = stub["uri"]
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            time.sleep(delay)
            occ = fetch_occupation(uri)
            if occ:
                all_occupations.append(occ)
                new += 1

        if verbose:
            print(f"{new} new occupations")

    if verbose:
        print(f"\n  Total unique occupations: {len(all_occupations)}")

    # Save occupations CSV
    occ_path = save_occupations(all_occupations, output_dir / "esco_occupations.csv")

    # Aggregate skills
    skills = aggregate_skills(all_occupations)
    save_skills(skills, output_dir / "esco_skills.csv")

    # Generate artifacts
    generate_artifacts(all_occupations, output_dir / "artifacts.jsonl")

    if verbose:
        print(f"  Unique skills: {len(skills)}")
        print("  Top 10 skills:")
        for s in skills[:10]:
            print(f"    [{s['occupation_count']:3d}] {s['title']}")

    # Manifest
    manifest = {
        "source": "ESCO (European Commission)",
        "api_base": _API_BASE,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "occupation_count": len(all_occupations),
        "unique_skills": len(skills),
        "content_hash": hashlib.sha256(
            occ_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if verbose:
        print(f"\n  Occupations: {occ_path}")
        print("  [OK] ESCO ingestion complete")

    return occ_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest ESCO occupations")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--max-per-query", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    run_ingestion(
        output_dir=args.output_dir,
        queries=args.queries,
        max_per_query=args.max_per_query,
        delay=args.delay,
    )
