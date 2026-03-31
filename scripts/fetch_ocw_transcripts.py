#!/usr/bin/env python3
"""Fetch supplementary text content for MIT OCW video courses.

For each course with has_video=true, fetches:
  1. Course main page (description)
  2. Syllabus page  (/pages/syllabus/)
  3. Calendar page  (/pages/calendar/)
  4. Video gallery page(s) to extract lecture titles

Saves enriched text to ocw_transcripts.jsonl.

Usage:
    python scripts/fetch_ocw_transcripts.py --max-courses 5   # test
    python scripts/fetch_ocw_transcripts.py                    # full run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

INPUT_PATH = Path("/Volumes/CORNADA_D/ADS/data/multimodal/ocw_video_metadata.jsonl")
OUTPUT_PATH = Path("/Volumes/CORNADA_D/ADS/data/multimodal/ocw_transcripts.jsonl")
BASE_URL = "https://ocw.mit.edu"
DELAY = 0.5  # seconds between requests
USER_AGENT = "ADS-Research/1.0 (academic; polite scraper; 0.5s delay)"


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------

class TextExtractor(HTMLParser):
    """Strips tags, keeps text.  Skips <script>, <style>, <nav>, <footer>."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._pieces)
        # collapse whitespace
        return re.sub(r"\s+", " ", raw).strip()


class LinkExtractor(HTMLParser):
    """Extracts <a href=...> links and their text from HTML."""

    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []  # (href, text)
        self._current_href: Optional[str] = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            self._current_href = href
            self._current_text = []

    def handle_data(self, data):
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href is not None:
            text = " ".join(self._current_text).strip()
            self.links.append((self._current_href, text))
            self._current_href = None
            self._current_text = []


def extract_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.get_text()


def extract_links(html: str) -> list[tuple[str, str]]:
    parser = LinkExtractor()
    parser.feed(html)
    return parser.links


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch URL, return HTML string or None on error."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  WARN fetch failed {url}: {exc}", file=sys.stderr)
        return None


def fetch_with_delay(url: str) -> Optional[str]:
    """Fetch + polite delay."""
    html = fetch_url(url)
    time.sleep(DELAY)
    return html


# ---------------------------------------------------------------------------
# Page-specific extractors
# ---------------------------------------------------------------------------

def extract_lecture_titles_from_links(links: list[tuple[str, str]]) -> list[str]:
    """Pull lecture/session titles from link text on calendar or video pages."""
    titles = []
    for href, text in links:
        text_clean = text.strip()
        if not text_clean:
            continue
        # Video gallery links often point to individual video pages
        if "/video_galleries/" in href or "/resources/" in href:
            # Skip generic nav links
            if text_clean.lower() in ("download course", "download", ""):
                continue
            titles.append(text_clean)
    return titles


class TableCellExtractor(HTMLParser):
    """Extract text from <td> and <th> cells, one list item per cell."""

    def __init__(self):
        super().__init__()
        self.cells: list[str] = []
        self._in_cell = False
        self._buf: list[str] = []
        self._skip = 0

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1
        if tag in ("td", "th"):
            self._in_cell = True
            self._buf = []

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip > 0:
            self._skip -= 1
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            text = re.sub(r"\s+", " ", " ".join(self._buf)).strip()
            if text:
                self.cells.append(text)

    def handle_data(self, data):
        if self._in_cell and self._skip == 0:
            self._buf.append(data)


def extract_calendar_lectures(html: str) -> list[str]:
    """Extract lecture titles from a calendar page.

    Calendar pages typically have a table with columns like:
      SES# | TOPICS | KEY DATES
    We extract table cells and look for lecture-like content.
    """
    parser = TableCellExtractor()
    parser.feed(html)

    lectures = []
    skip_words = {
        "topics", "key dates", "ses #", "ses#", "session", "lec #", "lec#",
        "homework", "problem set", "quiz", "exam", "due", "no class",
        "holiday", "readings", "assignments", "projects",
    }

    for cell in parser.cells:
        # Skip short cells (just numbers, dates, etc.)
        if len(cell) < 8:
            continue
        # Skip header-like cells
        if cell.lower().strip() in skip_words:
            continue
        # Skip cells that are just dates or assignment references
        if re.match(r"^(Homework|Problem set|Quiz|Exam|Due|No class)\b", cell, re.IGNORECASE):
            continue
        # Keep cells that look like topic descriptions
        # Must have some alphabetic content
        if sum(1 for c in cell if c.isalpha()) < 5:
            continue
        # Truncate to reasonable length (single topic, not a run-on)
        cell = cell[:150].strip()
        if cell not in lectures:
            lectures.append(cell)

    # Deduplicate while preserving order — also try regex on full text as fallback
    if not lectures:
        text = extract_text(html)
        for m in re.finditer(
            r"(?:Lecture|Session|Lec\.?|Class)\s*(\d+)\s*[:\-–]?\s*(.{5,100}?)(?=\s*(?:Lecture|Session|Lec\.?|Class)\s*\d|\s*$)",
            text, re.IGNORECASE
        ):
            title = f"Lecture {m.group(1)}: {m.group(2).strip()}"
            if title not in lectures:
                lectures.append(title)

    return lectures


def process_course(slug: str, course: Optional[dict] = None) -> dict:
    """Fetch enriched content for a single video course."""
    result = {
        "ocw_slug": slug,
        "has_transcript": False,
        "transcript_text": "",
        "lecture_titles": [],
        "syllabus_text": "",
        "calendar_text": "",
        "page_text_length": 0,
        "pages_fetched": [],
        "errors": [],
    }

    course_base = f"{BASE_URL}/courses/{slug}"
    all_text_parts: list[str] = []

    # 1. Main course page — description
    html = fetch_with_delay(course_base + "/")
    if html:
        result["pages_fetched"].append("main")
        main_text = extract_text(html)
        all_text_parts.append(main_text)

        # Find nav links for this course
        links = extract_links(html)
        nav_paths = {}
        for href, text in links:
            if f"/courses/{slug}/" in href:
                key = text.strip().lower()
                nav_paths[key] = href

        # 2. Syllabus
        syllabus_href = nav_paths.get("syllabus")
        if syllabus_href:
            syl_url = BASE_URL + syllabus_href if syllabus_href.startswith("/") else syllabus_href
            syl_html = fetch_with_delay(syl_url)
            if syl_html:
                result["pages_fetched"].append("syllabus")
                syl_text = extract_text(syl_html)
                result["syllabus_text"] = syl_text[:3000]
                all_text_parts.append(syl_text)

        # 3. Calendar
        calendar_href = nav_paths.get("calendar")
        if calendar_href:
            cal_url = BASE_URL + calendar_href if calendar_href.startswith("/") else calendar_href
            cal_html = fetch_with_delay(cal_url)
            if cal_html:
                result["pages_fetched"].append("calendar")
                cal_text = extract_text(cal_html)
                result["calendar_text"] = cal_text[:3000]
                all_text_parts.append(cal_text)

                # Extract structured lecture titles
                lectures = extract_calendar_lectures(cal_html)
                if lectures:
                    result["lecture_titles"].extend(lectures)

        # 4. Video gallery — get lecture titles from links
        video_hrefs = [
            href for href, text in links
            if "/video_galleries/" in href and f"/courses/{slug}/" in href
        ]
        for vhref in video_hrefs[:2]:  # at most 2 galleries
            vurl = BASE_URL + vhref if vhref.startswith("/") else vhref
            vhtml = fetch_with_delay(vurl)
            if vhtml:
                result["pages_fetched"].append("video_gallery")
                vlinks = extract_links(vhtml)
                # Video page links typically contain lecture titles
                # Generic nav/section labels to skip
                _nav_labels = {
                    "syllabus", "calendar", "readings", "assignments",
                    "exams", "download course", "lecture notes",
                    "related resources", "projects", "class videos",
                    "lecture videos", "recitation videos", "video lectures",
                    "course home", "course info", "recommend readings",
                    "recommended readings", "study guides",
                }
                for href2, text2 in vlinks:
                    text2 = text2.strip()
                    if not text2 or len(text2) < 6:
                        continue
                    # Must be a link within this course
                    if f"/courses/{slug}/" not in href2:
                        continue
                    # Must point to a video or resource page (not nav)
                    if not any(seg in href2 for seg in ("/video_galleries/", "/resources/")):
                        # also skip if it's a /pages/ nav link
                        if "/pages/" in href2:
                            continue
                    if text2.lower() in _nav_labels:
                        continue
                    # Skip if it's the course title itself
                    if course and text2 == course.get("course_title", ""):
                        continue
                    if text2 not in result["lecture_titles"]:
                        result["lecture_titles"].append(text2)
                vtext = extract_text(vhtml)
                all_text_parts.append(vtext)

        # 5. Check for transcript/caption links anywhere
        all_links = extract_links(html)
        for href, text in all_links:
            lower = (href + " " + text).lower()
            if any(kw in lower for kw in ("transcript", "caption", "subtitle", ".srt", ".vtt")):
                result["has_transcript"] = True
                break
    else:
        result["errors"].append("main_page_fetch_failed")

    # Assemble combined text
    combined = "\n\n".join(all_text_parts)
    result["transcript_text"] = combined[:2000]
    result["page_text_length"] = len(combined)

    # If we got lecture titles from calendar or video gallery, mark as enriched
    if result["lecture_titles"] or result["page_text_length"] > 500:
        result["has_transcript"] = True  # we have meaningful text content

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch OCW video course text content")
    parser.add_argument("--max-courses", type=int, default=0,
                        help="Limit number of courses (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip slugs already in output file")
    args = parser.parse_args()

    # Load input
    video_courses = []
    with open(INPUT_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("has_video"):
                video_courses.append(rec)

    print(f"Found {len(video_courses)} video courses in {INPUT_PATH}")

    # Resume support
    done_slugs: set[str] = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                r = json.loads(line)
                done_slugs.add(r["ocw_slug"])
        print(f"Resuming: {len(done_slugs)} already done")

    # Limit
    todo = [c for c in video_courses if c["ocw_slug"] not in done_slugs]
    if args.max_courses > 0:
        todo = todo[: args.max_courses]

    print(f"Will process {len(todo)} courses")

    mode = "a" if args.resume else "w"
    with open(OUTPUT_PATH, mode) as out:
        for i, course in enumerate(todo):
            slug = course["ocw_slug"]
            print(f"[{i+1}/{len(todo)}] {slug}")
            result = process_course(slug, course)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            n_lectures = len(result["lecture_titles"])
            n_pages = len(result["pages_fetched"])
            print(f"  -> pages={n_pages} lectures={n_lectures} text={result['page_text_length']} chars")

    # Summary
    stats = {"total": 0, "with_lectures": 0, "with_text_gt500": 0, "errors": 0}
    with open(OUTPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            stats["total"] += 1
            if r["lecture_titles"]:
                stats["with_lectures"] += 1
            if r["page_text_length"] > 500:
                stats["with_text_gt500"] += 1
            if r["errors"]:
                stats["errors"] += 1

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
