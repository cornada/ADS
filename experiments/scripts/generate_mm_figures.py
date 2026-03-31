"""Generate figures for ACM MM Dataset Track paper.

Figures:
1. Resource type distribution by department (stacked bar)
2. Resource richness distribution (histogram)
3. Cross-modal prediction results (bar chart)

Usage:
    python experiments/scripts/generate_mm_figures.py
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIGURE_DIR = PROJECT_ROOT / "ACM_MM_Dataset" / "Sources" / "figures"

# Color palette
COLORS = {
    "video": "#E74C3C",
    "notes": "#3498DB",
    "interactive": "#2ECC71",
    "assessment": "#F39C12",
    "bar": "#34495E",
    "highlight": "#E74C3C",
}


def load_data() -> List[Dict[str, Any]]:
    """Load multimodal metadata."""
    path = PROJECT_ROOT / "data" / "multimodal" / "ocw_video_metadata.jsonl"
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def fig1_resource_type_distribution(data: List[Dict]) -> None:
    """Stacked bar: resource type distribution by department."""
    # Group by department
    dept_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "video": 0, "notes": 0, "interactive": 0, "assessment": 0, "total": 0
    })

    for d in data:
        depts = d.get("department_numbers", [])
        dept = depts[0] if depts else "other"
        dept_stats[dept]["total"] += 1
        if d["has_video"]:
            dept_stats[dept]["video"] += 1
        if d["has_notes"]:
            dept_stats[dept]["notes"] += 1
        if d["has_interactive"]:
            dept_stats[dept]["interactive"] += 1
        if d["has_assessment"]:
            dept_stats[dept]["assessment"] += 1

    # Top 15 departments by count
    top_depts = sorted(dept_stats.keys(), key=lambda k: dept_stats[k]["total"], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(top_depts))
    width = 0.6

    bottoms = np.zeros(len(top_depts))
    for rtype, color in [
        ("video", COLORS["video"]),
        ("notes", COLORS["notes"]),
        ("interactive", COLORS["interactive"]),
        ("assessment", COLORS["assessment"]),
    ]:
        values = [dept_stats[d][rtype] for d in top_depts]
        ax.bar(x, values, width, bottom=bottoms, label=rtype.capitalize(), color=color, alpha=0.85)
        bottoms += np.array(values)

    ax.set_xlabel("MIT Department Number", fontsize=12)
    ax.set_ylabel("Number of Courses", fontsize=12)
    ax.set_title("Multimedia Resource Distribution by Department", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(top_depts, rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURE_DIR / "fig1_resource_distribution.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig2_richness_histogram(data: List[Dict]) -> None:
    """Histogram of resource richness scores."""
    richness = [d["resource_richness"] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))

    counts = Counter(richness)
    x_vals = sorted(counts.keys())
    y_vals = [counts[v] for v in x_vals]

    bars = ax.bar(x_vals, y_vals, color=COLORS["bar"], alpha=0.85, width=0.6)

    # Annotate bars
    for bar, count in zip(bars, y_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Resource Richness Score (0 = text only, 4 = all media types)", fontsize=11)
    ax.set_ylabel("Number of Courses", fontsize=11)
    ax.set_title("Distribution of Multimedia Resource Richness", fontsize=13)
    ax.set_xticks(range(5))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURE_DIR / "fig2_richness_histogram.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig3_video_by_level(data: List[Dict]) -> None:
    """Video availability by course level."""
    level_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"video": 0, "total": 0})

    for d in data:
        level = d.get("level", "Unknown") or "Unknown"
        level_stats[level]["total"] += 1
        if d["has_video"]:
            level_stats[level]["video"] += 1

    levels = sorted(level_stats.keys())
    totals = [level_stats[l]["total"] for l in levels]
    video_pcts = [level_stats[l]["video"] / max(level_stats[l]["total"], 1) * 100 for l in levels]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    x = np.arange(len(levels))
    bars = ax1.bar(x, totals, 0.4, color=COLORS["bar"], alpha=0.7, label="Total courses")

    ax2 = ax1.twinx()
    ax2.plot(x, video_pcts, "o-", color=COLORS["video"], linewidth=2, markersize=8, label="Video %")

    ax1.set_xlabel("Course Level", fontsize=11)
    ax1.set_ylabel("Number of Courses", fontsize=11)
    ax2.set_ylabel("Video Availability (%)", fontsize=11, color=COLORS["video"])
    ax1.set_title("Course Count and Video Availability by Level", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURE_DIR / "fig3_video_by_level.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig4_resource_type_cooccurrence(data: List[Dict]) -> None:
    """Heatmap of resource type co-occurrence."""
    types = ["has_video", "has_notes", "has_interactive", "has_assessment"]
    labels = ["Video", "Notes", "Interactive", "Assessment"]
    n = len(types)

    matrix = np.zeros((n, n))
    for d in data:
        for i in range(n):
            for j in range(n):
                if d[types[i]] and d[types[j]]:
                    matrix[i, j] += 1

    # Normalize by diagonal (proportion of type_i that also has type_j)
    diag = np.diag(matrix).copy()
    diag[diag == 0] = 1
    norm_matrix = matrix / diag[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            text = f"{norm_matrix[i, j]:.2f}"
            color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=11, color=color)

    ax.set_title("Resource Type Co-occurrence\n(P(col | row))", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = FIGURE_DIR / "fig4_cooccurrence.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    logger.info("Loaded %d OCW courses", len(data))

    # Filter to successfully fetched (non-empty resource types or has metadata)
    valid = [d for d in data if d.get("course_title")]
    logger.info("Valid courses (with metadata): %d", len(valid))

    fig1_resource_type_distribution(valid)
    fig2_richness_histogram(valid)
    fig3_video_by_level(valid)
    fig4_resource_type_cooccurrence(valid)

    logger.info("All figures generated in %s", FIGURE_DIR)


if __name__ == "__main__":
    main()
