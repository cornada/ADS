"""HeadHunter (hh.ru) vacancy ingestion for Russian labor market data.

Fetches job vacancies from the HH.ru public API to build a skill demand
profile for MISIS-relevant professional areas. Extracts key_skills,
salary ranges, and experience requirements.

API docs: https://github.com/hhru/api
Public endpoints (no auth): vacancies search, professional_roles, dictionaries

Rate limits: ~5 req/sec for anonymous access.

Usage:
    python -m ads_core.ingest.headhunter
    python -m ads_core.ingest.headhunter --roles 96,10,156 --per-role 100
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
_DEFAULT_DIR = _REPO_ROOT / "data" / "raw" / "headhunter"
_API_BASE = "https://api.hh.ru"

# MISIS-relevant professional role IDs from HH.ru
# Fetched from /professional_roles endpoint
MISIS_ROLE_IDS = {
    # IT
    "96": "Программист, разработчик",
    "10": "Аналитик",
    "12": "Арт-директор, креативный директор",
    "150": "DevOps-инженер",
    "165": "Data Engineer",
    "160": "Data Scientist",
    "104": "Системный администратор",
    "116": "Тестировщик",
    "114": "Системный инженер",
    "113": "Специалист по информационной безопасности",
    # Engineering / Production
    "55": "Инженер-конструктор, инженер-проектировщик",
    "57": "Инженер-технолог",
    "143": "Инженер по качеству",
    "144": "Инженер по эксплуатации",
    "82": "Научный специалист, исследователь",
    # Materials / Metallurgy (mapped by text search)
}

# Search queries for MISIS-specific domains
MISIS_SEARCH_QUERIES = [
    "материаловедение",
    "металлургия",
    "нанотехнологии",
    "горное дело",
    "информатика и вычислительная техника",
    "электроника",
    "энергетика",
    "экология промышленная",
    "аддитивные технологии",
    "DevOps",
    "machine learning инженер",
    "data scientist",
]

VACANCY_FIELDS = [
    "vacancy_id", "name", "salary_from", "salary_to", "salary_currency",
    "experience", "schedule", "employment", "area",
    "employer_name", "professional_roles", "key_skills",
    "published_at", "url",
]

SKILL_FIELDS = [
    "skill_name", "vacancy_count", "avg_salary_from", "avg_salary_to",
    "salary_currency", "search_query",
]


def _api_get(endpoint: str, params: Optional[Dict[str, str]] = None) -> Any:
    """Make a GET request to HH.ru API."""
    url = f"{_API_BASE}{endpoint}"
    if params:
        url = f"{url}?{urlencode(params)}"

    req = Request(url, headers={
        "User-Agent": "ADS-Research/1.0 (academic research)",
        "Accept": "application/json",
    })
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_vacancies_by_query(
    query: str,
    *,
    area: str = "113",  # Russia
    per_page: int = 100,
    max_pages: int = 5,
    delay: float = 0.3,
) -> List[Dict[str, Any]]:
    """Fetch vacancies matching a text query.

    Args:
        query: Search text.
        area: Area ID (113=Russia, 1=Moscow, 2=St.Petersburg).
        per_page: Results per page (max 100).
        max_pages: Maximum pages to fetch.
        delay: Delay between requests (rate limiting).

    Returns:
        List of vacancy dicts with key_skills from full vacancy fetch.
    """
    vacancies = []
    seen_ids = set()

    for page in range(max_pages):
        try:
            data = _api_get("/vacancies", {
                "text": query,
                "area": area,
                "per_page": str(per_page),
                "page": str(page),
                "only_with_salary": "false",
            })
        except (URLError, json.JSONDecodeError) as e:
            logger.warning("Failed to fetch page %d for '%s': %s", page, query, e)
            break

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            vid = item["id"]
            if vid in seen_ids:
                continue
            seen_ids.add(vid)

            # Fetch full vacancy for key_skills
            time.sleep(delay)
            try:
                full = _api_get(f"/vacancies/{vid}")
            except (URLError, json.JSONDecodeError):
                full = item  # fallback to search result

            salary = full.get("salary") or {}
            roles = [r.get("name", "") for r in full.get("professional_roles", [])]
            skills = [s["name"] for s in full.get("key_skills", [])]

            vacancies.append({
                "vacancy_id": vid,
                "name": full.get("name", ""),
                "salary_from": salary.get("from"),
                "salary_to": salary.get("to"),
                "salary_currency": salary.get("currency", ""),
                "experience": full.get("experience", {}).get("id", ""),
                "schedule": full.get("schedule", {}).get("id", ""),
                "employment": full.get("employment", {}).get("id", ""),
                "area": full.get("area", {}).get("name", ""),
                "employer_name": full.get("employer", {}).get("name", ""),
                "professional_roles": ",".join(roles),
                "key_skills": ",".join(skills),
                "published_at": full.get("published_at", ""),
                "url": full.get("alternate_url", ""),
                "search_query": query,
            })

        logger.info("Query '%s' page %d: %d items", query, page, len(items))

        if page >= data.get("pages", 0) - 1:
            break
        time.sleep(delay)

    return vacancies


def aggregate_skills(vacancies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate key_skills across vacancies to build a demand profile."""
    from collections import Counter, defaultdict

    skill_counts: Counter = Counter()
    skill_salaries: Dict[str, List] = defaultdict(list)
    skill_queries: Dict[str, set] = defaultdict(set)

    for v in vacancies:
        skills = [s.strip() for s in v.get("key_skills", "").split(",") if s.strip()]
        query = v.get("search_query", "")
        sal_from = v.get("salary_from")
        sal_to = v.get("salary_to")
        currency = v.get("salary_currency", "")

        for skill in skills:
            skill_counts[skill] += 1
            skill_queries[skill].add(query)
            if sal_from or sal_to:
                skill_salaries[skill].append((sal_from, sal_to, currency))

    rows = []
    for skill, count in skill_counts.most_common():
        sals = skill_salaries.get(skill, [])
        avg_from = None
        avg_to = None
        currency = ""
        if sals:
            froms = [s[0] for s in sals if s[0]]
            tos = [s[1] for s in sals if s[1]]
            currency = sals[0][2] if sals else ""
            avg_from = round(sum(froms) / len(froms)) if froms else None
            avg_to = round(sum(tos) / len(tos)) if tos else None

        rows.append({
            "skill_name": skill,
            "vacancy_count": count,
            "avg_salary_from": avg_from,
            "avg_salary_to": avg_to,
            "salary_currency": currency,
            "search_query": ",".join(sorted(skill_queries.get(skill, set()))),
        })

    return rows


def save_vacancies(vacancies: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save vacancies to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VACANCY_FIELDS + ["search_query"])
        writer.writeheader()
        writer.writerows(vacancies)
    return output_path


def save_skills(skills: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save aggregated skills to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SKILL_FIELDS)
        writer.writeheader()
        writer.writerows(skills)
    return output_path


def generate_artifacts(vacancies: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate ADS artifacts from HH vacancies.

    Each vacancy becomes a JOB_ROLE artifact, parallel to O*NET occupations
    but sourced from the live Russian labor market.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for v in vacancies:
            vid = v["vacancy_id"]
            name = v["name"]
            skills = v.get("key_skills", "")
            employer = v.get("employer_name", "")
            area = v.get("area", "")
            exp = v.get("experience", "")

            parts = [
                f"{name}.",
                f"Employer: {employer}." if employer else "",
                f"Location: {area}." if area else "",
                f"Experience: {exp}." if exp else "",
                f"Key skills: {skills}." if skills else "",
                "Source: HeadHunter (hh.ru).",
            ]
            text = " ".join(p for p in parts if p)

            aid = f"hh:{vid}"
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            rec = {
                "artifact_id": aid,
                "type": "JOB_ROLE",
                "source_url": v.get("url", ""),
                "source_hash": text_hash,
                "timestamp": now,
                "text": text,
                "metadata": {
                    "institution": "MARKET_RU",
                    "vacancy_id": vid,
                    "name": name,
                    "employer": employer,
                    "area": area,
                    "experience": exp,
                    "key_skills": skills.split(",") if skills else [],
                    "salary_from": v.get("salary_from"),
                    "salary_to": v.get("salary_to"),
                    "salary_currency": v.get("salary_currency", ""),
                    "search_query": v.get("search_query", ""),
                    "country": "RU",
                    "provenance": {
                        "source_url": v.get("url", ""),
                        "fetched_at": now,
                        "raw_text_hash": text_hash,
                        "dataset": "headhunter",
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
    max_per_query: int = 50,
    delay: float = 0.3,
    verbose: bool = True,
) -> Path:
    """Run HeadHunter vacancy ingestion.

    Args:
        output_dir: Where to write output.
        queries: Search queries. Defaults to MISIS_SEARCH_QUERIES.
        max_per_query: Max vacancies per query.
        delay: API rate limiting delay.
        verbose: Print progress.

    Returns:
        Path to vacancies CSV.
    """
    if output_dir is None:
        output_dir = _DEFAULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = MISIS_SEARCH_QUERIES

    if verbose:
        print("HeadHunter Vacancy Ingestion")
        print(f"  Queries: {len(queries)}")
        print(f"  Max per query: {max_per_query}")

    all_vacancies: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for query in queries:
        if verbose:
            print(f"  Fetching: '{query}'...", end=" ", flush=True)

        max_pages = max(1, max_per_query // 100 + 1)
        results = fetch_vacancies_by_query(
            query,
            per_page=min(100, max_per_query),
            max_pages=max_pages,
            delay=delay,
        )

        # Deduplicate across queries
        new = 0
        for v in results:
            if v["vacancy_id"] not in seen_ids:
                seen_ids.add(v["vacancy_id"])
                all_vacancies.append(v)
                new += 1

        if verbose:
            print(f"{new} new vacancies")

    if verbose:
        print(f"\n  Total unique vacancies: {len(all_vacancies)}")

    # Save vacancies
    vac_path = save_vacancies(all_vacancies, output_dir / "hh_vacancies.csv")

    # Aggregate skills
    skills = aggregate_skills(all_vacancies)
    skills_path = save_skills(skills, output_dir / "hh_skills.csv")

    # Generate artifacts
    generate_artifacts(all_vacancies, output_dir / "artifacts.jsonl")

    if verbose:
        print(f"  Artifacts: {len(all_vacancies)}")
        print(f"  Unique skills: {len(skills)}")
        print("  Top 10 skills:")
        for s in skills[:10]:
            sal = ""
            if s["avg_salary_from"]:
                sal = f" ({s['avg_salary_from']}-{s['avg_salary_to']} {s['salary_currency']})"
            print(f"    [{s['vacancy_count']:3d}] {s['skill_name']}{sal}")

    # Manifest
    manifest = {
        "source": "HeadHunter (hh.ru) Public API",
        "api_base": _API_BASE,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "vacancy_count": len(all_vacancies),
        "unique_skills": len(skills),
        "content_hash": hashlib.sha256(
            vac_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"\n  Vacancies: {vac_path}")
        print(f"  Skills: {skills_path}")
        print("  [OK] HeadHunter ingestion complete")

    return vac_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest HeadHunter vacancies")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--max-per-query", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    run_ingestion(
        output_dir=args.output_dir,
        queries=args.queries,
        max_per_query=args.max_per_query,
        delay=args.delay,
    )
