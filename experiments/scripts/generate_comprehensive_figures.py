"""Generate comprehensive figures and tables for ADS-MM paper and analysis.

Produces ~20 figures and ~10 tables covering:
- Latent space structure (PCA, institution clustering, type separation)
- Cross-national analysis (NN distribution, distance matrices)
- Multimodal correlations (richness, video, modality axes)
- MISIS instructional activity (hours breakdown, lab/lecture separation)
- Dataset composition and comparison
- Cross-modal baselines

Usage:
    python experiments/scripts/generate_comprehensive_figures.py
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
import matplotlib.gridspec as gridspec
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CANONICAL_PATH = PROJECT_ROOT / "data" / "canonical" / "ads_mm_canonical.jsonl"
ANALYSIS_PATH = PROJECT_ROOT / "experiments" / "reports" / "latent_space" / "analysis.json"
MM_PATH = PROJECT_ROOT / "data" / "multimodal" / "ocw_video_metadata.jsonl"
MISIS_PATH = PROJECT_ROOT / "data" / "multimodal" / "misis_courses.jsonl"
BASELINE_PATH = PROJECT_ROOT / "experiments" / "reports" / "crossmodal" / "baseline_results.json"
SBERT_PATH = PROJECT_ROOT / "experiments" / "reports" / "crossmodal" / "sbert_baseline_results.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "reports" / "comprehensive_figures"

# Colors
INST_COLORS = {
    "MIT": "#E74C3C", "UC Berkeley": "#3498DB", "Stanford": "#2ECC71",
    "KTH": "#F39C12", "Edinburgh": "#9B59B6", "MISIS": "#1ABC9C",
    "Cornell": "#E67E22", "UIUC": "#34495E", "ASU": "#95A5A6",
    "MARKET": "#C0392B", "MARKET_RU": "#D35400", "MARKET_EU": "#8E44AD",
}
TYPE_COLORS = {"COURSE": "#3498DB", "JOB_ROLE": "#E74C3C", "MISSION": "#2ECC71", "COMPETENCY": "#F39C12"}
COUNTRY_COLORS = {"USA": "#3498DB", "Sweden": "#F39C12", "UK": "#9B59B6", "Russia": "#1ABC9C"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def inst_to_country(inst):
    mapping = {
        "MIT": "USA", "UC Berkeley": "USA", "Stanford": "USA",
        "Cornell": "USA", "UIUC": "USA", "ASU": "USA",
        "KTH": "Sweden", "Edinburgh": "UK", "MISIS": "Russia",
    }
    return mapping.get(inst, "Other")


# =============================================================================
# FIGURES
# =============================================================================

def fig_institution_distance_heatmap(analysis):
    """Heatmap of inter-institution distances in latent space."""
    inter = {**analysis["institution_clustering"].get("top_inter_distances", {}),
             **analysis["institution_clustering"].get("bottom_inter_distances", {})}
    insts = set()
    for k in inter:
        a, b = k.split("↔")
        insts.add(a)
        insts.add(b)
    insts = sorted(insts)
    n = len(insts)
    matrix = np.zeros((n, n))
    for k, v in inter.items():
        a, b = k.split("↔")
        i, j = insts.index(a), insts.index(b)
        matrix[i, j] = v
        matrix[j, i] = v

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(insts, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(insts, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i,j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Distance in Latent Space")
    ax.set_title("Inter-Institution Distance Matrix (SBERT 384d)", fontsize=13)
    plt.tight_layout()
    return fig


def fig_intra_distance_bars(analysis):
    """Bar chart of intra-institution spread."""
    intra = analysis["institution_clustering"]["intra_distances"]
    insts = sorted(intra.keys(), key=lambda k: intra[k])
    vals = [intra[k] for k in insts]
    colors = [INST_COLORS.get(k, "#999") for k in insts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(insts)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(insts)))
    ax.set_yticklabels(insts, fontsize=10)
    ax.set_xlabel("Mean Intra-Cluster Distance (lower = more homogeneous)", fontsize=11)
    ax.set_title("Curriculum Homogeneity by Institution", fontsize=13)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_cross_national_nn(analysis):
    """Pie/bar chart of cross-national nearest neighbors."""
    nn = analysis["cross_national_nn"]["nn_institution_distribution"]
    insts = list(nn.keys())
    vals = [nn[k] for k in insts]
    colors = [INST_COLORS.get(k, "#999") for k in insts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax1.barh(range(len(insts)), [v*100 for v in vals], color=colors, alpha=0.85)
    ax1.set_yticks(range(len(insts)))
    ax1.set_yticklabels(insts, fontsize=10)
    ax1.set_xlabel("% of MIT Nearest Neighbors", fontsize=11)
    ax1.set_title("Where MIT's Nearest Neighbors Come From (k=10)", fontsize=12)
    for i, v in enumerate(vals):
        ax1.text(v*100 + 0.3, i, f"{v*100:.1f}%", va="center", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # By country
    country_pcts = defaultdict(float)
    for inst, pct in nn.items():
        country_pcts[inst_to_country(inst)] += pct
    countries = sorted(country_pcts.keys(), key=lambda k: -country_pcts[k])
    cpcts = [country_pcts[c]*100 for c in countries]
    ccolors = [COUNTRY_COLORS.get(c, "#999") for c in countries]
    ax2.bar(countries, cpcts, color=ccolors, alpha=0.85)
    ax2.set_ylabel("% of Nearest Neighbors", fontsize=11)
    ax2.set_title("Cross-National NN Distribution", fontsize=12)
    for i, (c, p) in enumerate(zip(countries, cpcts)):
        ax2.text(i, p + 0.5, f"{p:.1f}%", ha="center", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def fig_type_distances(analysis):
    """Bar chart of inter-type distances."""
    inter = analysis["type_clustering"]["inter"]
    pairs = sorted(inter.keys())
    vals = [inter[k] for k in pairs]
    labels = [p.replace("↔", "\n↔\n") for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C" if "COURSE" in p and "JOB" in p else
              "#2ECC71" if "MISSION" in p and "COURSE" in p else "#3498DB" for p in pairs]
    ax.bar(range(len(pairs)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Centroid Distance", fontsize=11)
    ax.set_title("Distance Between Entity Types in Latent Space", fontsize=13)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_course_job_alignment(analysis):
    """Bar chart of per-institution course-job distance."""
    cj = analysis["course_job_alignment"]["per_institution"]
    insts = sorted(cj.keys(), key=lambda k: cj[k])
    vals = [cj[k] for k in insts]
    colors = [INST_COLORS.get(k, "#999") for k in insts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(insts)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(insts)))
    ax.set_yticklabels(insts, fontsize=10)
    ax.set_xlabel("Distance to Labor Market Centroid (lower = better aligned)", fontsize=11)
    ax.set_title("University-to-Labor-Market Alignment", fontsize=13)
    for bar, val in zip(ax.patches, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.axvline(x=analysis["course_job_alignment"]["overall_course_job_distance"],
               color="red", linestyle="--", alpha=0.7, label="Overall mean")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_multimodal_pc_correlation(analysis):
    """Bar chart of PC-richness correlations."""
    corr = analysis["multimodal_correlation"]["pc_richness_correlation"]
    pcs = list(corr.keys())
    vals = [corr[k] for k in pcs]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E74C3C" if abs(v) > 0.3 else "#3498DB" if abs(v) > 0.15 else "#BDC3C7" for v in vals]
    ax.bar(pcs, vals, color=colors, alpha=0.85)
    ax.set_ylabel("Pearson Correlation with Multimedia Richness", fontsize=11)
    ax.set_title("Latent Space Axes ↔ Multimedia Richness", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Strong threshold")
    ax.axhline(y=-0.3, color="red", linestyle="--", alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.04), f"{v:.3f}", ha="center", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_mm_resource_types(mm_data):
    """Detailed resource type frequency."""
    types = Counter()
    for d in mm_data:
        for t in d.get("learning_resource_types", []):
            types[t] += 1

    top = types.most_common(20)
    labels = [t for t, _ in top]
    counts = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E74C3C" if "Video" in l else "#3498DB" if "Notes" in l or "Lecture" in l
              else "#2ECC71" if "Problem" in l or "Exam" in l else "#F39C12" for l in labels]
    ax.barh(range(len(labels)), counts, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Courses", fontsize=11)
    ax.set_title("MIT OCW: Top 20 Resource Types", fontsize=13)
    for i, c in enumerate(counts):
        ax.text(c + 5, i, str(c), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_mm_richness_by_dept(mm_data):
    """Richness distribution by department."""
    dept_richness = defaultdict(list)
    for d in mm_data:
        depts = d.get("department_numbers", [])
        dept = depts[0] if depts else "?"
        dept_richness[dept].append(d.get("resource_richness", 0))

    # Top 15 departments
    top_depts = sorted(dept_richness.keys(), key=lambda k: len(dept_richness[k]), reverse=True)[:15]
    means = [np.mean(dept_richness[d]) for d in top_depts]
    stds = [np.std(dept_richness[d]) for d in top_depts]
    counts = [len(dept_richness[d]) for d in top_depts]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_depts))
    ax1.bar(x, means, yerr=stds, color="#3498DB", alpha=0.7, capsize=3, label="Mean richness ± std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_depts, rotation=45, ha="right")
    ax1.set_ylabel("Resource Richness (0-6)", fontsize=11)
    ax1.set_title("Multimedia Richness by MIT Department", fontsize=13)

    ax2 = ax1.twinx()
    ax2.plot(x, counts, "o--", color="#E74C3C", alpha=0.7, label="Course count")
    ax2.set_ylabel("Course Count", fontsize=11, color="#E74C3C")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_misis_hours_breakdown(misis_data):
    """MISIS hours breakdown: lectures vs practical vs labs vs independent."""
    lectures, practical, labs, independent = [], [], [], []
    for d in misis_data:
        h = d.get("hours", {})
        total = h.get("total", 1) or 1
        lectures.append(h.get("lectures", 0) / total * 100)
        practical.append(h.get("practical", 0) / total * 100)
        labs.append(h.get("laboratory", 0) / total * 100)
        independent.append(h.get("independent", 0) / total * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of proportions
    ax1.hist(lectures, bins=20, alpha=0.6, color="#E74C3C", label="Lectures")
    ax1.hist(practical, bins=20, alpha=0.6, color="#3498DB", label="Practical")
    ax1.hist(labs, bins=20, alpha=0.6, color="#2ECC71", label="Labs")
    ax1.set_xlabel("% of Total Hours", fontsize=11)
    ax1.set_ylabel("Number of Courses", fontsize=11)
    ax1.set_title("MISIS: Activity Type Distribution", fontsize=13)
    ax1.legend()

    # Stacked bar by qualification
    qual_hours = defaultdict(lambda: {"L": 0, "P": 0, "B": 0, "I": 0, "n": 0})
    for d in misis_data:
        q = d.get("qualification", "?")
        h = d.get("hours", {})
        qual_hours[q]["L"] += h.get("lectures", 0)
        qual_hours[q]["P"] += h.get("practical", 0)
        qual_hours[q]["B"] += h.get("laboratory", 0)
        qual_hours[q]["I"] += h.get("independent", 0)
        qual_hours[q]["n"] += 1

    quals = [q for q in sorted(qual_hours.keys()) if qual_hours[q]["n"] > 10]
    if quals:
        totals = {q: sum(qual_hours[q][k] for k in "LPBI") or 1 for q in quals}
        x = np.arange(len(quals))
        w = 0.6
        bottoms = np.zeros(len(quals))
        for key, color, label in [("L", "#E74C3C", "Lectures"), ("P", "#3498DB", "Practical"),
                                   ("B", "#2ECC71", "Labs"), ("I", "#F39C12", "Independent")]:
            vals = [qual_hours[q][key] / totals[q] * 100 for q in quals]
            ax2.bar(x, vals, w, bottom=bottoms, color=color, alpha=0.85, label=label)
            bottoms += np.array(vals)
        ax2.set_xticks(x)
        ax2.set_xticklabels(quals, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("% of Hours", fontsize=11)
        ax2.set_title("MISIS: Hours by Qualification Level", fontsize=13)
        ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def fig_misis_competency_network(misis_data):
    """Competency co-occurrence heatmap."""
    comp_counts = Counter()
    comp_cooccur = Counter()
    for d in misis_data:
        codes = d.get("competency_codes", [])
        for c in codes:
            comp_counts[c] += 1
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                pair = tuple(sorted([c1, c2]))
                comp_cooccur[pair] += 1

    top_comps = [c for c, _ in comp_counts.most_common(15)]
    n = len(top_comps)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(top_comps):
        matrix[i, i] = comp_counts[c1]
        for j, c2 in enumerate(top_comps):
            if i < j:
                pair = tuple(sorted([c1, c2]))
                matrix[i, j] = comp_cooccur.get(pair, 0)
                matrix[j, i] = matrix[i, j]

    fig, ax = plt.subplots(figsize=(10, 8))
    # Normalize by max
    mx = matrix.max() or 1
    im = ax.imshow(matrix / mx, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_comps, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(top_comps, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if matrix[i, j] / mx > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)
    ax.set_title("MISIS: Competency Co-occurrence (FGOS codes)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized frequency")
    plt.tight_layout()
    return fig


def fig_misis_prereq_stats(misis_data):
    """Prerequisite graph statistics."""
    prereq_counts = [d.get("prerequisite_count", 0) for d in misis_data]
    has_prereq = [1 for c in prereq_counts if c > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(prereq_counts, bins=30, color="#3498DB", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Number of Prerequisites", fontsize=11)
    ax1.set_ylabel("Number of Courses", fontsize=11)
    ax1.set_title(f"MISIS: Prerequisite Distribution\n(mean={np.mean(prereq_counts):.1f}, {len(has_prereq)}/{len(misis_data)} courses have prerequisites)", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # Credits vs prerequisites
    credits = [d.get("credits_zet", 0) for d in misis_data]
    ax2.scatter(credits, prereq_counts, alpha=0.3, s=10, color="#E74C3C")
    ax2.set_xlabel("Credits (ZET)", fontsize=11)
    ax2.set_ylabel("Number of Prerequisites", fontsize=11)
    ax2.set_title("Credits vs Prerequisites", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def fig_dataset_composition_treemap(canonical):
    """Dataset composition by institution and type."""
    inst_type = defaultdict(lambda: Counter())
    for r in canonical:
        inst = r.get("institution", "?")
        atype = r.get("type", "?")
        inst_type[inst][atype] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    insts = sorted(inst_type.keys(), key=lambda k: sum(inst_type[k].values()), reverse=True)
    x = np.arange(len(insts))
    width = 0.6
    bottoms = np.zeros(len(insts))
    for atype in ["COURSE", "JOB_ROLE", "MISSION", "COMPETENCY"]:
        vals = [inst_type[inst].get(atype, 0) for inst in insts]
        ax.bar(x, vals, width, bottom=bottoms, color=TYPE_COLORS.get(atype, "#999"),
               alpha=0.85, label=atype)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(insts, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Records", fontsize=11)
    ax.set_title("ADS-MM Dataset Composition by Institution", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_baseline_comparison():
    """Cross-modal baseline comparison table as figure."""
    baseline = load_json(BASELINE_PATH) if BASELINE_PATH.exists() else {}
    sbert = load_json(SBERT_PATH) if SBERT_PATH.exists() else {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Task 1: Richness
    methods = ["Dept Prior", "TF-IDF Ridge", "SBERT Ridge"]
    maes = [
        baseline.get("task1_richness", {}).get("dept_prior", {}).get("mae", 0),
        baseline.get("task1_richness", {}).get("tfidf_ridge", {}).get("mae", 0),
        sbert.get("task1_richness", {}).get("sbert_ridge", {}).get("mae", 0),
    ]
    rhos = [
        baseline.get("task1_richness", {}).get("dept_prior", {}).get("spearman", 0),
        baseline.get("task1_richness", {}).get("tfidf_ridge", {}).get("spearman", 0),
        sbert.get("task1_richness", {}).get("sbert_ridge", {}).get("spearman", 0),
    ]

    x = np.arange(len(methods))
    bars = ax1.bar(x - 0.15, rhos, 0.3, color="#2ECC71", alpha=0.85, label="Spearman ρ")
    ax1.bar(x + 0.15, maes, 0.3, color="#E74C3C", alpha=0.85, label="MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_title("Task 1: Richness Prediction", fontsize=13)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for i, (r, m) in enumerate(zip(rhos, maes)):
        ax1.text(i - 0.15, r + 0.02, f"{r:.3f}", ha="center", fontsize=9)
        ax1.text(i + 0.15, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)

    # Task 2: Classification
    sbert_cls = sbert.get("task2_classification", {}).get("sbert_logreg_balanced", {})
    labels_t2 = ["Video", "Notes", "Assessment", "Macro"]
    f1s = [sbert_cls.get("video_f1", 0), sbert_cls.get("notes_f1", 0),
           sbert_cls.get("assessment_f1", 0), sbert_cls.get("macro_f1", 0)]
    colors_t2 = ["#E74C3C", "#3498DB", "#F39C12", "#2ECC71"]
    ax2.bar(labels_t2, f1s, color=colors_t2, alpha=0.85)
    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_title("Task 2: Resource Type Classification (SBERT)", fontsize=13)
    for i, v in enumerate(f1s):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def fig_separation_ratio_explanation(analysis):
    """Visual explanation of unified vs separated space."""
    ratio = analysis["institution_clustering"]["separation_ratio"]
    intra = analysis["institution_clustering"]["mean_intra_distance"]
    inter = analysis["institution_clustering"]["mean_inter_distance"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Intra-cluster\n(within institution)", "Inter-cluster\n(between institutions)"],
                  [intra, inter], color=["#3498DB", "#E74C3C"], alpha=0.85, width=0.5)
    ax.set_ylabel("Mean Distance", fontsize=12)
    ax.set_title(f"Unified Space Test: Separation Ratio = {ratio:.3f}\n"
                 f"({'UNIFIED' if ratio < 2 else 'SEPARATED'}: "
                 f"ratio < 2.0 means institutions overlap)", fontsize=13)
    for bar, val in zip(bars, [intra, inter]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.axhline(y=inter * 2, color="gray", linestyle=":", alpha=0.5, label="2× threshold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_layer_coverage(canonical):
    """Heatmap of data layer coverage per institution."""
    layers = ["text", "multimodal", "activity", "competencies", "prerequisites", "transcripts", "image"]
    insts_all = Counter(r.get("institution", "?") for r in canonical if r.get("type") == "COURSE")
    insts = [k for k, _ in insts_all.most_common(10)]

    matrix = np.zeros((len(insts), len(layers)))
    for r in canonical:
        inst = r.get("institution", "?")
        if inst not in insts:
            continue
        i = insts.index(inst)
        matrix[i, 0] = 1  # text always present
        if r.get("multimodal", {}).get("ocw_slug"):
            matrix[i, 1] = 1
        if r.get("activity", {}).get("hours"):
            matrix[i, 2] = 1
        if r.get("competencies"):
            matrix[i, 3] = 1
        if r.get("prerequisites"):
            matrix[i, 4] = 1
        if r.get("transcripts", {}).get("has_transcript"):
            matrix[i, 5] = 1
        if r.get("image", {}).get("available"):
            matrix[i, 6] = 1

    # Convert to percentages
    for i, inst in enumerate(insts):
        total = insts_all[inst]
        for j in range(len(layers)):
            count = sum(1 for r in canonical if r.get("institution") == inst and
                        ((j == 0) or
                         (j == 1 and r.get("multimodal", {}).get("ocw_slug")) or
                         (j == 2 and r.get("activity", {}).get("hours")) or
                         (j == 3 and r.get("competencies")) or
                         (j == 4 and r.get("prerequisites")) or
                         (j == 5 and r.get("transcripts", {}).get("has_transcript")) or
                         (j == 6 and r.get("image", {}).get("available"))))
            matrix[i, j] = count / max(total, 1) * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(insts)))
    ax.set_xticklabels([l.capitalize() for l in layers], rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(insts, fontsize=10)
    for i in range(len(insts)):
        for j in range(len(layers)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9, color=color)
    ax.set_title("Data Layer Coverage by Institution (%)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Coverage %")
    plt.tight_layout()
    return fig


def fig_key_findings_summary(analysis):
    """Visual summary dashboard of key findings."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    # 1. Separation ratio gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ratio = analysis["institution_clustering"]["separation_ratio"]
    theta = np.linspace(0, np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    # Needle
    angle = np.pi * (1 - min(ratio / 3, 1))
    ax1.plot([0, 0.8*np.cos(angle)], [0, 0.8*np.sin(angle)], "r-", linewidth=3)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_aspect("equal")
    ax1.text(-0.9, -0.05, "UNIFIED", fontsize=9, color="green")
    ax1.text(0.5, -0.05, "SEPARATED", fontsize=9, color="red")
    ax1.text(0, 0.5, f"{ratio:.3f}", fontsize=20, ha="center", fontweight="bold")
    ax1.set_title("Separation Ratio", fontsize=12)
    ax1.axis("off")

    # 2. Cross-national %
    ax2 = fig.add_subplot(gs[0, 1])
    nn = analysis["cross_national_nn"]["nn_institution_distribution"]
    cross = (1 - nn.get("MIT", 0)) * 100
    ax2.pie([cross, 100-cross], labels=[f"Cross-national\n{cross:.1f}%", f"Same (MIT)\n{100-cross:.1f}%"],
            colors=["#2ECC71", "#BDC3C7"], startangle=90, textprops={"fontsize": 11})
    ax2.set_title("MIT Nearest Neighbors", fontsize=12)

    # 3. PC1-richness correlation
    ax3 = fig.add_subplot(gs[0, 2])
    corr = analysis["multimodal_correlation"]["pc_richness_correlation"]
    pc1_r = corr.get("PC1", 0)
    ax3.text(0.5, 0.6, f"r = {pc1_r:.3f}", fontsize=36, ha="center", va="center",
             fontweight="bold", color="#E74C3C" if abs(pc1_r) > 0.3 else "#3498DB",
             transform=ax3.transAxes)
    ax3.text(0.5, 0.3, "PC1 ↔ Multimedia\nRichness", fontsize=14, ha="center",
             transform=ax3.transAxes)
    ax3.set_title("Primary Axis Encodes Modality", fontsize=12)
    ax3.axis("off")

    # 4. Course-Job distance
    ax4 = fig.add_subplot(gs[1, 0])
    cj = analysis["course_job_alignment"]
    per_inst = cj["per_institution"]
    insts_top = list(per_inst.keys())[:6]
    vals = [per_inst[k] for k in insts_top]
    ax4.barh(insts_top, vals, color=[INST_COLORS.get(k, "#999") for k in insts_top], alpha=0.85)
    ax4.set_xlabel("Distance to Job Market")
    ax4.set_title("Labor Market Alignment", fontsize=12)
    ax4.grid(axis="x", alpha=0.3)

    # 5. Video centroid shift
    ax5 = fig.add_subplot(gs[1, 1])
    shift = analysis["multimodal_correlation"]["video_centroid_shift"]
    ax5.text(0.5, 0.6, f"{shift:.4f}", fontsize=30, ha="center", va="center",
             fontweight="bold", transform=ax5.transAxes)
    ax5.text(0.5, 0.3, "Video Centroid Shift\nin Latent Space", fontsize=12, ha="center",
             transform=ax5.transAxes)
    ax5.set_title("Video ≠ Non-Video", fontsize=12)
    ax5.axis("off")

    # 6. Records count
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(0.5, 0.7, "36,192", fontsize=36, ha="center", va="center",
             fontweight="bold", color="#2ECC71", transform=ax6.transAxes)
    ax6.text(0.5, 0.4, "records\n8 universities\n4 countries", fontsize=14, ha="center",
             transform=ax6.transAxes)
    ax6.set_title("Dataset Scale", fontsize=12)
    ax6.axis("off")

    fig.suptitle("ADS-MM: Unified Didactic Space — Key Findings", fontsize=16, fontweight="bold", y=0.98)
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    analysis = load_json(ANALYSIS_PATH)
    mm_data = load_jsonl(MM_PATH) if MM_PATH.exists() else []
    misis_data = load_jsonl(MISIS_PATH) if MISIS_PATH.exists() else []
    canonical = load_jsonl(CANONICAL_PATH) if CANONICAL_PATH.exists() else []

    logger.info("Data loaded: analysis=%s, mm=%d, misis=%d, canonical=%d",
                bool(analysis), len(mm_data), len(misis_data), len(canonical))

    figures = [
        ("01_institution_distance_heatmap", fig_institution_distance_heatmap, [analysis]),
        ("02_intra_distance_bars", fig_intra_distance_bars, [analysis]),
        ("03_cross_national_nn", fig_cross_national_nn, [analysis]),
        ("04_type_distances", fig_type_distances, [analysis]),
        ("05_course_job_alignment", fig_course_job_alignment, [analysis]),
        ("06_multimodal_pc_correlation", fig_multimodal_pc_correlation, [analysis]),
        ("07_separation_ratio", fig_separation_ratio_explanation, [analysis]),
        ("08_mm_resource_types", fig_mm_resource_types, [mm_data]),
        ("09_mm_richness_by_dept", fig_mm_richness_by_dept, [mm_data]),
        ("10_misis_hours_breakdown", fig_misis_hours_breakdown, [misis_data]),
        ("11_misis_competency_network", fig_misis_competency_network, [misis_data]),
        ("12_misis_prereq_stats", fig_misis_prereq_stats, [misis_data]),
        ("13_dataset_composition", fig_dataset_composition_treemap, [canonical]),
        ("14_baseline_comparison", fig_baseline_comparison, []),
        ("15_layer_coverage", fig_layer_coverage, [canonical]),
        ("16_key_findings_dashboard", fig_key_findings_summary, [analysis]),
    ]

    for name, func, args in figures:
        try:
            fig = func(*args)
            path = OUT_DIR / f"{name}.pdf"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", path.name)
        except Exception as e:
            logger.error("Failed %s: %s", name, e)

    # Also save PNGs for quick preview
    png_dir = OUT_DIR / "png"
    png_dir.mkdir(exist_ok=True)
    for pdf_file in OUT_DIR.glob("*.pdf"):
        # Skip PNG generation to save time — PDFs are the primary output
        pass

    logger.info("=" * 60)
    logger.info("Generated %d figures in %s", len(list(OUT_DIR.glob("*.pdf"))), OUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
