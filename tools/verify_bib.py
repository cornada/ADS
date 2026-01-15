#!/usr/bin/env python3
"""
verify_bib.py — BibTeX reference verifier (Crossref/OpenAlex/arXiv)

Design goals:
- deterministic output
- conservative matching (avoid false positives)
- produce a JSON report + exit non-zero when unverified refs exist (CI gate)

Usage:
  python tools/verify_bib.py path/to/references.bib --report reports/reference_audit.json --fail_on_unverified

NOTE: this script requires internet access to query Crossref/OpenAlex/arXiv APIs.
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import urllib.parse
import urllib.request

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)

# --------------------------
# Minimal BibTeX parsing
# --------------------------
ENTRY_START = re.compile(r"@\w+\s*{\s*([^,]+)\s*,", re.IGNORECASE)

def parse_bibtex(path: str) -> List[Dict[str, Any]]:
    """
    Minimal parser good enough for standard .bib files.
    Returns list of dicts: {key, raw, fields: {name: value}}
    """
    txt = open(path, "r", encoding="utf-8").read()
    entries: List[Dict[str, Any]] = []
    # split by '@' but keep it
    parts = re.split(r"(?=@\w+\s*{)", txt)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = ENTRY_START.search(part)
        if not m:
            continue
        key = m.group(1).strip()
        fields: Dict[str, str] = {}
        # naive field extraction: name = {...} or "..."
        for fm in re.finditer(r"\n\s*([a-zA-Z_]+)\s*=\s*({(?:[^{}]|{[^{}]*})*}|\"(?:\\\"|[^\"])*\")\s*,?", part):
            name = fm.group(1).lower()
            val = fm.group(2).strip()
            # strip braces/quotes
            if val.startswith("{") and val.endswith("}"):
                val = val[1:-1].strip()
            elif val.startswith("\"") and val.endswith("\""):
                val = val[1:-1].strip()
            fields[name] = val
        entries.append({"key": key, "raw": part, "fields": fields})
    return entries

# --------------------------
# API helpers
# --------------------------
def http_get_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ADS-ref-verifier/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)

def crossref_lookup_doi(doi: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    try:
        j = http_get_json(url)
        return j.get("message")
    except Exception:
        return None

def crossref_search(title: str, year: Optional[str], first_author: Optional[str]) -> List[Dict[str, Any]]:
    # Conservative: search by title; filter by year if given
    q = title
    params = {"query.title": q, "rows": "5"}
    if year:
        # Crossref filter syntax: from-pub-date / until-pub-date
        params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    try:
        j = http_get_json(url)
        return j.get("message", {}).get("items", []) or []
    except Exception:
        return []

def openalex_search(title: str, year: Optional[str]) -> List[Dict[str, Any]]:
    params = {"search": title, "per-page": "5"}
    if year:
        params["filter"] = f"from_publication_date:{year}-01-01,to_publication_date:{year}-12-31"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    try:
        j = http_get_json(url)
        return j.get("results", []) or []
    except Exception:
        return []

def arxiv_lookup(arxiv_id: str) -> Optional[Dict[str, Any]]:
    # arXiv API is Atom; to keep deps minimal, do a lightweight regex parse.
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ADS-ref-verifier/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
        # Extract <title> ... </title> of entry (first is feed title, second is paper title)
        titles = re.findall(r"<title>(.*?)</title>", xml, flags=re.DOTALL)
        if len(titles) >= 2:
            paper_title = re.sub(r"\s+", " ", titles[1]).strip()
        else:
            paper_title = None
        # Extract published year
        pub = re.search(r"<published>(\d{4})-\d{2}-\d{2}T", xml)
        pub_year = pub.group(1) if pub else None
        return {"title": paper_title, "year": pub_year, "raw": xml[:2000]}
    except Exception:
        return None

# --------------------------
# Matching heuristics
# --------------------------
def norm_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def title_similarity(a: str, b: str) -> float:
    # simple token Jaccard (conservative)
    sa = set(norm_title(a).split())
    sb = set(norm_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def extract_first_author(author_field: Optional[str]) -> Optional[str]:
    if not author_field:
        return None
    # BibTeX "and" separated
    parts = [p.strip() for p in author_field.split(" and ") if p.strip()]
    if not parts:
        return None
    # "Last, First" or "First Last"
    a = parts[0]
    if "," in a:
        return a.split(",")[0].strip()
    return a.split()[-1].strip()

# --------------------------
# Main verification
# --------------------------
def verify_entry(entry: Dict[str, Any], sleep_s: float = 0.0) -> Dict[str, Any]:
    key = entry["key"]
    f = entry["fields"]
    title = f.get("title") or ""
    year = f.get("year")
    author = extract_first_author(f.get("author"))

    raw = entry["raw"]
    doi = f.get("doi")
    if not doi:
        m = DOI_RE.search(raw)
        doi = m.group(0) if m else None
    arxiv = f.get("eprint")
    if not arxiv:
        m2 = ARXIV_RE.search(raw)
        arxiv = m2.group(1) if m2 else None

    report: Dict[str, Any] = {
        "key": key,
        "status": "unverified",
        "input": {"title": title, "year": year, "author": author, "doi": doi, "arxiv": arxiv},
        "crossref": None,
        "openalex": None,
        "arxiv": None,
        "candidates": [],
        "notes": [],
    }

    # 1) DOI path
    if doi:
        msg = crossref_lookup_doi(doi)
        if msg and msg.get("title"):
            cr_title = (msg.get("title") or [""])[0] if isinstance(msg.get("title"), list) else msg.get("title")
            sim = title_similarity(title, cr_title) if title and cr_title else 0.0
            report["crossref"] = {"doi": doi, "title": cr_title, "year": (msg.get("published-print") or msg.get("published-online") or {}).get("date-parts")}
            if sim >= 0.7 or not title:
                report["status"] = "verified"
            else:
                report["status"] = "mismatch"
                report["notes"].append(f"DOI metadata title mismatch (sim={sim:.2f})")
        else:
            report["notes"].append("DOI lookup failed")
        if sleep_s:
            time.sleep(sleep_s)
        return report

    # 2) arXiv path
    if arxiv:
        ax = arxiv_lookup(arxiv)
        report["arxiv"] = ax
        if ax and ax.get("title"):
            sim = title_similarity(title, ax["title"]) if title else 0.0
            if sim >= 0.7 or not title:
                report["status"] = "verified"
            else:
                report["status"] = "mismatch"
                report["notes"].append(f"arXiv title mismatch (sim={sim:.2f})")
        else:
            report["notes"].append("arXiv lookup failed")
        if sleep_s:
            time.sleep(sleep_s)
        return report

    # 3) Search path (Crossref + OpenAlex) — candidates only, require manual confirmation
    if not title:
        report["notes"].append("missing title field")
        return report

    cr_items = crossref_search(title, year, author)
    for it in cr_items:
        it_title = (it.get("title") or [""])[0] if isinstance(it.get("title"), list) else it.get("title")
        sim = title_similarity(title, it_title) if it_title else 0.0
        cand = {"source": "crossref", "doi": it.get("DOI"), "title": it_title, "year": it.get("issued", {}).get("date-parts"), "sim": sim}
        if sim >= 0.7:
            report["candidates"].append(cand)

    oa_items = openalex_search(title, year)
    for it in oa_items:
        it_title = it.get("title") or ""
        sim = title_similarity(title, it_title) if it_title else 0.0
        cand = {"source": "openalex", "id": it.get("id"), "doi": it.get("doi"), "title": it_title, "year": it.get("publication_year"), "sim": sim}
        if sim >= 0.7:
            report["candidates"].append(cand)

    if report["candidates"]:
        report["status"] = "candidate_found"
        report["notes"].append("candidates found; add DOI and rerun for auto-verify")
    else:
        report["notes"].append("no candidates found")
    if sleep_s:
        time.sleep(sleep_s)
    return report

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bib", help="Path to .bib file")
    ap.add_argument("--report", default="reports/reference_audit.json", help="Output JSON report path")
    ap.add_argument("--fail_on_unverified", action="store_true", help="Exit non-zero if any entry is not verified")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between API calls (seconds)")
    args = ap.parse_args()

    entries = parse_bibtex(args.bib)
    reports: List[Dict[str, Any]] = []
    for e in entries:
        reports.append(verify_entry(e, sleep_s=args.sleep))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"bib": args.bib, "n": len(reports), "entries": reports}, f, indent=2, ensure_ascii=False)

    verified = sum(1 for r in reports if r["status"] == "verified")
    unverified = [r for r in reports if r["status"] != "verified"]

    print(f"[verify_bib] total={len(reports)} verified={verified} unverified={len(unverified)}")
    if args.fail_on_unverified and unverified:
        print("[verify_bib] FAIL: some entries are not verified. See report:", args.report)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
