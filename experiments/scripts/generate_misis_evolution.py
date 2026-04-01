#!/usr/bin/env python3
"""MISIS Curriculum Evolution Analysis (2021-2025) — All Five Years.

Analyzes how MISIS courses and programs evolved across all 5 academic years,
using the Agent-Didactic Spaces embedding framework to reveal:
- Program composition changes across all 5 years
- Course embedding drift in the Didactic Space (all available years)
- Objective score trajectories (market/mission/university alignment)
- Competency evolution patterns (2024→2025 where data is richest)
- Course lifecycle classification (new/stable/modified/extinct)
- Multi-year trajectory tracking for courses spanning 3+ years

Data quality by year:
  2021: 172 courses (names + direction, no descriptions)
  2022:  51 courses (names + direction, no descriptions)
  2023:  17 courses (names only, minimal)
  2024: 1302 courses (rich: names + goals + content + prereqs + credits)
  2025:  921 courses (rich: names + goals + prereqs + credits)

Produces 8 figures + self-contained HTML report.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
BASE = Path("/Users/aleksandrvolkov/Desktop/ADS")
DATA = BASE / "data/raw/misis"
PARSED = DATA / "parsed"
UNIFIED = BASE / "data/processed/unified_v5/artifacts.jsonl"
OUT_DIR = BASE / "experiments/reports/misis_evolution"
FIG_DIR = OUT_DIR / "figures"

ALL_YEARS = [2021, 2022, 2023, 2024, 2025]

DPI = 300
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.weight': 'bold', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 15,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'figure.titleweight': 'bold', 'legend.fontsize': 10,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
})
LEGEND_PROP = {'weight': 'bold', 'size': 10}

YEAR_COLORS = {
    2021: '#2166AC', 2022: '#67A9CF', 2023: '#D1E5F0',
    2024: '#F4A582', 2025: '#B2182B',
}
YEAR_MARKERS = {2021: 'o', 2022: 's', 2023: 'D', 2024: '^', 2025: 'v'}


# ═══════════════════════════════════════════════════════════
# Layer 1: Data Loading
# ═══════════════════════════════════════════════════════════

def load_catalog():
    with open(DATA / "catalog.csv") as f:
        return list(csv.DictReader(f))

def load_parsed_year(year):
    path = PARSED / f"{year}_courses.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get('course_name', '').strip()]

def load_all_years():
    """Load parsed courses for all years."""
    all_courses = {}
    for year in ALL_YEARS:
        courses = load_parsed_year(year)
        all_courses[year] = courses
        print(f"  {year}: {len(courses)} courses")
    return all_courses

def load_competencies():
    with open(DATA / "competencies.csv") as f:
        return list(csv.DictReader(f))

def load_temporal_snapshots():
    with open(DATA / "temporal_snapshots.csv") as f:
        return list(csv.DictReader(f))

def load_target_artifacts():
    targets = {'JOB_ROLE': [], 'MISSION': [], 'COMPETENCY': [], 'COURSE': []}
    with open(UNIFIED) as f:
        for line in f:
            a = json.loads(line)
            t = a.get('type', '')
            if t in targets:
                targets[t].append(a)
    return targets


def build_course_text(row):
    """Build embeddable text from a parsed course row."""
    parts = []
    name = row.get('course_name', '').strip()
    if name:
        parts.append(name)
    dept = row.get('department', '').strip()
    if dept:
        parts.append(f"Department: {dept}")
    fgos_dir = row.get('fgos_direction', '').strip()
    if fgos_dir:
        parts.append(f"Direction: {fgos_dir}")
    profile = row.get('profile', '').strip()
    if profile:
        parts.append(f"Profile: {profile}")
    goals = row.get('goals', '').strip()
    if goals:
        parts.append(goals[:500])
    content = row.get('content_summary', '').strip()
    if content and not goals:
        parts.append(content[:500])
    prereqs = row.get('prerequisites', '').strip()
    if prereqs:
        parts.append(f"Prerequisites: {prereqs[:200]}")
    return ". ".join(parts) if parts else ""


def build_course_key(row):
    """Build a stable key for matching across years: (fgos_code, normalized_name)."""
    fgos = row.get('fgos_code', '').strip()
    name = row.get('course_name', '').strip()
    # Normalize: lowercase, strip underscores/extra spaces
    name_norm = name.lower().replace('_', ' ').replace('  ', ' ').strip()
    return (fgos, name_norm)


def build_multi_year_index(all_courses):
    """Build an index of courses across all years.

    Returns dict: course_key -> {year: [row, ...]}
    """
    index = defaultdict(lambda: defaultdict(list))
    for year, courses in all_courses.items():
        for r in courses:
            key = build_course_key(r)
            if key[1]:  # non-empty name
                index[key][year].append(r)

    # Deduplicate: keep the row with longest text per year
    deduped = {}
    for key, year_dict in index.items():
        deduped[key] = {}
        for year, rows in year_dict.items():
            best = max(rows, key=lambda r: len(build_course_text(r)))
            deduped[key][year] = best
    return deduped


# ═══════════════════════════════════════════════════════════
# Layer 2: Embedding & Scoring
# ═══════════════════════════════════════════════════════════

def load_encoder():
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"  Loading encoder: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def encode_texts(model, texts, desc=""):
    print(f"  Encoding {len(texts)} texts{f' ({desc})' if desc else ''}...")
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 50,
                        batch_size=64)
    return vecs

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def build_centroid(vecs):
    m = np.mean(vecs, axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-8 else m


# ═══════════════════════════════════════════════════════════
# Layer 3: Analysis Functions
# ═══════════════════════════════════════════════════════════

def analyze_program_landscape(catalog):
    programs_per_year = defaultdict(set)
    program_names = {}
    program_levels = {}
    for row in catalog:
        y = int(row['year'])
        code = row['program_code']
        programs_per_year[y].add(code)
        program_names[code] = row['program_name']
        program_levels[code] = row.get('level', '?')

    years = sorted(programs_per_year.keys())
    all_programs = sorted(set().union(*programs_per_year.values()))

    matrix = np.zeros((len(all_programs), len(years)), dtype=int)
    for j, y in enumerate(years):
        for i, p in enumerate(all_programs):
            if p in programs_per_year[y]:
                matrix[i, j] = 1

    return {
        'years': years,
        'programs': all_programs,
        'names': program_names,
        'levels': program_levels,
        'matrix': matrix,
        'per_year': {y: len(programs_per_year[y]) for y in years},
    }


def analyze_multi_year_trajectories(course_index, embeddings_by_year):
    """Analyze courses that appear in 2+ years."""
    trajectories = []
    for key, year_dict in course_index.items():
        years_present = sorted(year_dict.keys())
        if len(years_present) < 2:
            continue
        # Check if we have embeddings for these years
        emb_years = [y for y in years_present if y in embeddings_by_year
                     and key in embeddings_by_year[y]]
        if len(emb_years) < 2:
            continue
        traj = {
            'key': key,
            'name': year_dict[years_present[0]].get('course_name', ''),
            'fgos': key[0],
            'years': emb_years,
            'embeddings': {y: embeddings_by_year[y][key] for y in emb_years},
            'texts': {y: build_course_text(year_dict[y]) for y in emb_years},
        }
        # Compute pairwise similarities between consecutive years
        sims = []
        for i in range(len(emb_years) - 1):
            y1, y2 = emb_years[i], emb_years[i+1]
            s = cosine_sim(traj['embeddings'][y1], traj['embeddings'][y2])
            sims.append((y1, y2, s))
        traj['consecutive_sims'] = sims
        # Total drift: first year to last year
        traj['total_drift_sim'] = cosine_sim(
            traj['embeddings'][emb_years[0]], traj['embeddings'][emb_years[-1]])
        trajectories.append(traj)
    return trajectories


def analyze_competency_evolution(competencies):
    comp_type_by_year = defaultdict(lambda: defaultdict(int))
    fgos_comp_year = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    unique_codes_by_year = defaultdict(set)

    for row in competencies:
        y = row['year']
        code = row['competency_code']
        ctype = row['competency_type']
        fgos = row['fgos_code']
        comp_type_by_year[y][ctype] += 1
        fgos_comp_year[fgos][y][ctype] += 1
        unique_codes_by_year[y].add(code)

    return {
        'type_by_year': dict(comp_type_by_year),
        'fgos_comp_year': dict(fgos_comp_year),
        'unique_codes_by_year': {y: len(s) for y, s in unique_codes_by_year.items()},
    }


def analyze_credit_changes(course_index):
    """Credit changes for courses appearing in 2024 and 2025."""
    changes = []
    for key, year_dict in course_index.items():
        if 2024 not in year_dict or 2025 not in year_dict:
            continue
        c24 = year_dict[2024].get('credits_zet', '').strip()
        c25 = year_dict[2025].get('credits_zet', '').strip()
        if c24 and c25:
            try:
                changes.append({
                    'name': year_dict[2024]['course_name'],
                    'fgos': key[0],
                    'cr_2024': float(c24),
                    'cr_2025': float(c25),
                    'delta': float(c25) - float(c24),
                })
            except ValueError:
                pass
    return changes


# ═══════════════════════════════════════════════════════════
# Layer 4: Figure Generation
# ═══════════════════════════════════════════════════════════

def fig_program_landscape(landscape):
    """F1: Program existence heatmap across all 5 years."""
    fig, ax = plt.subplots(figsize=(8, max(10, len(landscape['programs']) * 0.22)))

    matrix = landscape['matrix']
    years = landscape['years']
    programs = landscape['programs']
    names = landscape['names']
    levels = landscape['levels']

    presence_count = matrix.sum(axis=1)
    order = np.argsort(-presence_count)
    matrix = matrix[order]
    programs = [programs[i] for i in order]

    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    labels = [f"{p} {'[G]' if levels.get(p,'') == 'G' else '[U]'} {names.get(p, '')[:28]}"
              for p in programs]
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontweight='bold')
    ax.set_xlabel("Academic Year")
    ax.set_title("F1: MISIS Program Landscape (2021–2025)")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(programs)))
    ax2.set_yticklabels([str(int(presence_count[order[i]])) for i in range(len(programs))],
                        fontsize=6, color='#666')
    ax2.set_ylabel("Years present", fontsize=10)

    plt.tight_layout()
    path = FIG_DIR / "F1_program_landscape.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


def fig_multi_year_drift(trajectories):
    """F2: Multi-year embedding drift in PCA space."""
    from sklearn.decomposition import PCA

    # Collect all embeddings with year labels
    all_vecs = []
    all_labels = []
    all_years_list = []
    traj_indices = []  # (traj_idx, year)

    for ti, traj in enumerate(trajectories):
        for y in traj['years']:
            all_vecs.append(traj['embeddings'][y])
            all_labels.append(traj['name'][:40])
            all_years_list.append(y)
            traj_indices.append((ti, y))

    if len(all_vecs) < 3:
        print("  Skipping F2: too few multi-year vectors")
        return None

    all_vecs = np.array(all_vecs)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(all_vecs)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw trajectories (arrows between consecutive years of same course)
    for ti, traj in enumerate(trajectories):
        years = traj['years']
        # Find projected positions for this trajectory
        positions = {}
        for j, (idx, y) in enumerate(traj_indices):
            if idx == ti:
                positions[y] = proj[j]

        # Draw connecting arrows
        for i_y in range(len(years) - 1):
            y1, y2 = years[i_y], years[i_y + 1]
            if y1 in positions and y2 in positions:
                p1, p2 = positions[y1], positions[y2]
                drift = np.linalg.norm(p2 - p1)
                alpha = min(0.8, 0.2 + drift * 3)
                ax.annotate("", xy=p2, xytext=p1,
                             arrowprops=dict(arrowstyle="->", color='#555',
                                             lw=0.8, alpha=alpha))

    # Plot points colored by year
    for y in ALL_YEARS:
        mask = [i for i, yl in enumerate(all_years_list) if yl == y]
        if mask:
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=YEAR_COLORS[y], s=40, alpha=0.7,
                       marker=YEAR_MARKERS[y], label=str(y),
                       edgecolors='white', linewidths=0.5, zorder=5)

    # Label courses with longest trajectories (3+ years)
    long_trajs = [t for t in trajectories if len(t['years']) >= 3]
    for traj in sorted(long_trajs, key=lambda t: t['total_drift_sim'])[:10]:
        # Label at the midpoint of first and last year
        first_y = traj['years'][0]
        last_y = traj['years'][-1]
        for j, (idx, y) in enumerate(traj_indices):
            if idx == trajectories.index(traj) and y == last_y:
                ax.annotate(f"{traj['name'][:30]} ({first_y}→{last_y})",
                            proj[j], fontsize=6, alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.15',
                                      facecolor='yellow', alpha=0.5))
                break

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("F2: Multi-Year Course Trajectories in Didactic Space")
    ax.legend(prop=LEGEND_PROP, ncol=5, loc='upper center',
              bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = FIG_DIR / "F2_multi_year_drift.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


def fig_objective_trajectories(trajectories, centroids, obj_names):
    """F3: Market × Mission objective trajectories across all years."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for traj in trajectories:
        years = traj['years']
        if len(years) < 2:
            continue

        # Score against market and mission for each year
        market_scores = []
        mission_scores = []
        for y in years:
            emb = traj['embeddings'][y]
            market_scores.append(cosine_sim(emb, centroids['market']))
            mission_scores.append(cosine_sim(emb, centroids['mission']))

        # Draw trajectory
        for i in range(len(years) - 1):
            ax.annotate("", xy=(market_scores[i+1], mission_scores[i+1]),
                         xytext=(market_scores[i], mission_scores[i]),
                         arrowprops=dict(arrowstyle="->", color='#555',
                                         lw=0.7, alpha=0.35))

        # Plot points
        for i, y in enumerate(years):
            ax.scatter(market_scores[i], mission_scores[i],
                       c=YEAR_COLORS[y], s=30, alpha=0.6,
                       marker=YEAR_MARKERS[y], edgecolors='white',
                       linewidths=0.5, zorder=5)

    # Add year legend manually
    for y in ALL_YEARS:
        ax.scatter([], [], c=YEAR_COLORS[y], marker=YEAR_MARKERS[y],
                   s=60, label=str(y), edgecolors='white', linewidths=0.5)

    ax.set_xlabel("Market alignment")
    ax.set_ylabel("Mission alignment")
    ax.set_title("F3: Objective Trajectories Across Years (Market × Mission)")
    ax.legend(prop=LEGEND_PROP, ncol=5)

    plt.tight_layout()
    path = FIG_DIR / "F3_objective_trajectories.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


def fig_competency_evolution(comp_analysis):
    """F4: Competency type distribution changes."""
    type_by_year = comp_analysis['type_by_year']
    years = sorted(type_by_year.keys())
    comp_types = sorted(set().union(*(set(v.keys()) for v in type_by_year.values())))

    rich_years = [y for y in years if sum(type_by_year[y].values()) > 10]
    if len(rich_years) < 2:
        # Fall back: show all years that have data
        rich_years = [y for y in years if sum(type_by_year[y].values()) > 0]
    if len(rich_years) < 1:
        print("  Skipping F4: no competency data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(comp_types))
    width = 0.8 / max(len(rich_years), 1)
    for idx, y in enumerate(rich_years):
        counts = [type_by_year[y].get(ct, 0) for ct in comp_types]
        color = YEAR_COLORS.get(int(y), '#999')
        axes[0].bar(x + idx * width, counts, width, label=y, color=color, alpha=0.8)

    axes[0].set_xticks(x + width * len(rich_years) / 2)
    axes[0].set_xticklabels(comp_types, fontweight='bold')
    axes[0].set_ylabel("Competency mappings")
    axes[0].set_title("Competency type distribution by year")
    axes[0].legend(prop=LEGEND_PROP)

    # Right: per-FGOS heatmap for latest rich year
    fgos_data = comp_analysis['fgos_comp_year']
    fgos_totals = {f: sum(sum(yv.values()) for yv in yd.values())
                   for f, yd in fgos_data.items()}
    top_fgos = sorted(fgos_totals, key=fgos_totals.get, reverse=True)[:12]

    latest = rich_years[-1]
    matrix = np.zeros((len(top_fgos), len(comp_types)))
    for i, f in enumerate(top_fgos):
        for j, ct in enumerate(comp_types):
            matrix[i, j] = fgos_data.get(f, {}).get(latest, {}).get(ct, 0)

    im = axes[1].imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_yticks(range(len(top_fgos)))
    axes[1].set_yticklabels(top_fgos, fontsize=9)
    axes[1].set_xticks(range(len(comp_types)))
    axes[1].set_xticklabels(comp_types, fontweight='bold')
    axes[1].set_title(f"Per-FGOS competency profile ({latest})")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle("F4: Competency Evolution", fontweight='bold', fontsize=16)
    plt.tight_layout()
    path = FIG_DIR / "F4_competency_evolution.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


def fig_credit_redistribution(credit_changes):
    """F5: Credit changes for matched courses."""
    if not credit_changes:
        print("  Skipping F5: no credit data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cr24 = [c['cr_2024'] for c in credit_changes]
    cr25 = [c['cr_2025'] for c in credit_changes]
    deltas = [c['delta'] for c in credit_changes]

    axes[0].scatter(cr24, cr25, c='#1B9E77', alpha=0.5, s=40, edgecolors='white', linewidths=0.5)
    lim = max(max(cr24), max(cr25)) + 1
    axes[0].plot([0, lim], [0, lim], 'k--', alpha=0.3, label='No change')
    axes[0].set_xlabel("Credits 2024 (ЗЕТ)")
    axes[0].set_ylabel("Credits 2025 (ЗЕТ)")
    axes[0].set_title("Credit comparison")
    axes[0].legend(prop=LEGEND_PROP)

    axes[1].hist(deltas, bins=30, color='#F4A582', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1].axvline(x=np.mean(deltas), color='#B2182B', linestyle='-', alpha=0.7,
                    label=f'Mean: {np.mean(deltas):+.1f}')
    axes[1].set_xlabel("Credit change (2025 − 2024)")
    axes[1].set_ylabel("Number of courses")
    axes[1].set_title("Credit redistribution")
    axes[1].legend(prop=LEGEND_PROP)

    fig.suptitle("F5: Credit Redistribution 2024→2025", fontweight='bold', fontsize=16)
    plt.tight_layout()
    path = FIG_DIR / "F5_credit_redistribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


def fig_course_lifecycle(lifecycle_stats):
    """F6: Course lifecycle classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: 2024↔2025 lifecycle
    cats = ['Continuous', 'New in 2025', 'Left after 2024']
    counts = [lifecycle_stats['matched_24_25'],
              lifecycle_stats['only_2025'],
              lifecycle_stats['only_2024']]
    colors = ['#2166AC', '#1B9E77', '#F4A582']
    bars = axes[0].bar(cats, counts, color=colors, edgecolor='white', linewidth=2)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(count), ha='center', va='bottom', fontweight='bold', fontsize=13)
    axes[0].set_ylabel("Number of courses")
    axes[0].set_title("2024↔2025 Lifecycle")

    # Right: longevity distribution (how many years does each course span)
    longevity = lifecycle_stats['longevity_dist']
    years_span = sorted(longevity.keys())
    counts_span = [longevity[y] for y in years_span]
    colors_span = ['#D1E5F0', '#67A9CF', '#2166AC', '#F4A582', '#B2182B'][:len(years_span)]
    bars = axes[1].bar([f"{y} years" for y in years_span], counts_span,
                       color=colors_span[:len(years_span)], edgecolor='white', linewidth=2)
    for bar, count in zip(bars, counts_span):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(count), ha='center', va='bottom', fontweight='bold', fontsize=13)
    axes[1].set_ylabel("Number of courses")
    axes[1].set_title("Course longevity (years present)")

    fig.suptitle("F6: Course Lifecycle Classification", fontweight='bold', fontsize=16)
    plt.tight_layout()
    path = FIG_DIR / "F6_course_lifecycle.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


def fig_similarity_distribution(trajectories):
    """F7: Cosine similarity distribution for 2024→2025 matched courses."""
    sims_24_25 = []
    labels = []
    for traj in trajectories:
        if 2024 in traj['years'] and 2025 in traj['years']:
            s = cosine_sim(traj['embeddings'][2024], traj['embeddings'][2025])
            sims_24_25.append(s)
            labels.append(traj['name'])

    if not sims_24_25:
        print("  Skipping F7: no 2024-2025 pairs")
        return None

    sims = np.array(sims_24_25)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(sims, bins=40, color='#67A9CF', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(sims), color='#B2182B', linestyle='-', lw=2,
               label=f'Mean: {np.mean(sims):.3f}')
    ax.axvline(x=np.median(sims), color='#2166AC', linestyle='--', lw=2,
               label=f'Median: {np.median(sims):.3f}')
    p10 = np.percentile(sims, 10)
    ax.axvline(x=p10, color='orange', linestyle=':', lw=1.5,
               label=f'P10: {p10:.3f} (high drift)')

    ax.set_xlabel("Cosine similarity (2024 vs 2025 embedding)")
    ax.set_ylabel("Number of courses")
    ax.set_title("F7: Embedding Stability Distribution (2024→2025)")
    ax.legend(prop=LEGEND_PROP)

    plt.tight_layout()
    path = FIG_DIR / "F7_similarity_distribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path, sims, labels


def fig_objective_radar(trajectories, centroids, obj_names):
    """F8: Radar chart of mean objective scores per year."""
    # Compute mean scores per year
    year_scores = defaultdict(lambda: defaultdict(list))
    for traj in trajectories:
        for y in traj['years']:
            emb = traj['embeddings'][y]
            for obj in obj_names:
                year_scores[y][obj].append(cosine_sim(emb, centroids[obj]))

    if not year_scores:
        print("  Skipping F8: no data")
        return None

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    n_obj = len(obj_names)
    angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles += angles[:1]

    for y in sorted(year_scores.keys()):
        means = [np.mean(year_scores[y][o]) for o in obj_names]
        means += means[:1]
        color = YEAR_COLORS.get(y, '#999')
        ax.plot(angles, means, 'o-', color=color, linewidth=2, markersize=7,
                label=str(y), alpha=0.8)
        ax.fill(angles, means, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([o.capitalize() for o in obj_names], fontweight='bold', fontsize=12)
    ax.set_title("F8: Objective Profile by Year\n(mean over matched courses)", pad=20,
                 fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', prop=LEGEND_PROP)

    plt.tight_layout()
    path = FIG_DIR / "F8_objective_radar_yearly.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════
# Layer 5: HTML Report
# ═══════════════════════════════════════════════════════════

def img_to_base64(path):
    import base64
    if path is None or not Path(path).exists():
        return ""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def generate_html_report(results):
    figs = results['figures']
    stats = results['stats']
    drifters = results.get('top_drifters', [])
    anchors = results.get('top_anchors', [])
    obj_shift = results.get('objective_shift', {})
    long_trajs = results.get('long_trajectories', [])

    def fig_block(title, key, description):
        b64 = img_to_base64(figs.get(key))
        if not b64:
            return ""
        return f"""
        <div class="figure-block">
            <h3>{title}</h3>
            <p class="fig-desc">{description}</p>
            <img src="{b64}" alt="{title}">
        </div>"""

    def table_from_items(items, title, columns):
        if not items:
            return ""
        hdr = "".join(f"<th>{c}</th>" for c in columns)
        rows_html = ""
        for d in items:
            cells = "".join(f"<td>{d.get(c.lower().replace(' ','_').replace('δ','delta'), '?')}</td>"
                            for c in columns)
            rows_html += f"<tr>{cells}</tr>"
        return f"""<h3>{title}</h3>
        <table><tr>{hdr}</tr>{rows_html}</table>"""

    # Format drifter/anchor rows
    def fmt_items(items):
        if not items:
            return ""
        rows_html = ""
        for d in items:
            rows_html += f"""<tr>
                <td>{d['name'][:50]}</td>
                <td>{d['fgos']}</td>
                <td>{d['years_str']}</td>
                <td>{d['cos_sim']:.3f}</td>
                <td>{d.get('market_delta', 0):+.3f}</td>
                <td>{d.get('mission_delta', 0):+.3f}</td>
            </tr>"""
        return rows_html

    # Long trajectories table
    long_rows = ""
    for t in long_trajs[:20]:
        sims_str = " → ".join(f"{s:.3f}" for _, _, s in t['consecutive_sims'])
        long_rows += f"""<tr>
            <td>{t['name'][:45]}</td>
            <td>{t['fgos']}</td>
            <td>{' → '.join(str(y) for y in t['years'])}</td>
            <td>{t['total_drift_sim']:.3f}</td>
            <td>{sims_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>MISIS Curriculum Evolution (2021–2025)</title>
<style>
:root {{
    --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff;
    --card: #161b22; --border: #30363d; --red: #f85149;
    --green: #3fb950; --yellow: #d29922; --blue: #388bfd;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: var(--accent); font-size: 2rem; margin-bottom: 0.5rem;
     border-bottom: 2px solid var(--border); padding-bottom: 1rem; }}
h2 {{ color: var(--accent); font-size: 1.5rem; margin: 2rem 0 1rem;
     border-left: 4px solid var(--accent); padding-left: 0.8rem; }}
h3 {{ color: var(--fg); font-size: 1.2rem; margin: 1.5rem 0 0.8rem; }}
p {{ margin-bottom: 1rem; }}
.subtitle {{ color: #8b949e; font-size: 0.95rem; margin-bottom: 2rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 1rem; margin: 1.5rem 0; }}
.metric {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 1.2rem; text-align: center; }}
.metric .value {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
.metric .label {{ font-size: 0.8rem; color: #8b949e; margin-top: 0.3rem; }}
.figure-block {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
                padding: 1.5rem; margin: 1.5rem 0; }}
.figure-block img {{ width: 100%; border-radius: 4px; margin-top: 1rem; }}
.fig-desc {{ color: #8b949e; font-size: 0.9rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--card);
        font-size: 0.9rem; }}
th, td {{ padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: #1c2128; color: var(--accent); font-weight: 600; }}
tr:hover {{ background: #1c2128; }}
.insight {{ background: linear-gradient(135deg, #1a2332 0%, #162030 100%);
           border: 1px solid var(--blue); border-radius: 8px; padding: 1.2rem; margin: 1.5rem 0; }}
.insight h4 {{ color: var(--blue); margin-bottom: 0.5rem; }}
.year-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; margin: 1rem 0; }}
.year-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 6px;
             padding: 0.8rem; text-align: center; }}
.year-card .yr {{ font-size: 1.4rem; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">

<h1>MISIS Curriculum Evolution in the Didactic Space</h1>
<p class="subtitle">
    Agent-Didactic Spaces | Five-year temporal analysis of NUST MISIS (2021–2025)<br>
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
    Encoder: paraphrase-multilingual-MiniLM-L12-v2 (384d)
</p>

<h2>1. Data Coverage</h2>
<p>We analyze MISIS educational programs across 5 academic years. Data richness varies:
2024–2025 have full course-level parsing (goals, competencies, prerequisites),
while 2021–2023 have program-level metadata and course names.</p>
<div class="year-grid">
    <div class="year-card"><div class="yr" style="color:#2166AC">2021</div><div>{stats['courses_per_year'].get(2021, 0)} courses<br>{stats['programs_per_year'].get(2021, 0)} programs</div></div>
    <div class="year-card"><div class="yr" style="color:#67A9CF">2022</div><div>{stats['courses_per_year'].get(2022, 0)} courses<br>{stats['programs_per_year'].get(2022, 0)} programs</div></div>
    <div class="year-card"><div class="yr" style="color:#D1E5F0">2023</div><div>{stats['courses_per_year'].get(2023, 0)} courses<br>{stats['programs_per_year'].get(2023, 0)} programs</div></div>
    <div class="year-card"><div class="yr" style="color:#F4A582">2024</div><div>{stats['courses_per_year'].get(2024, 0)} courses<br>{stats['programs_per_year'].get(2024, 0)} programs</div></div>
    <div class="year-card"><div class="yr" style="color:#B2182B">2025</div><div>{stats['courses_per_year'].get(2025, 0)} courses<br>{stats['programs_per_year'].get(2025, 0)} programs</div></div>
</div>

<div class="metrics">
    <div class="metric"><div class="value">{stats['total_unique_courses']}</div><div class="label">Unique courses (all years)</div></div>
    <div class="metric"><div class="value">{stats['multi_year_courses']}</div><div class="label">Courses in 2+ years</div></div>
    <div class="metric"><div class="value">{stats['total_programs']}</div><div class="label">Total programs</div></div>
    <div class="metric"><div class="value">{stats.get('mean_cos_sim_24_25', 0):.3f}</div><div class="label">Mean cos sim (2024→2025)</div></div>
</div>

<h2>2. Program Landscape</h2>
<p>{stats['total_programs']} educational programs operated across 2021–2025.
The heatmap reveals which programs persisted and which were added or discontinued.</p>
{fig_block("Program Existence Heatmap", "F1",
           "Each row is a program (FGOS code + name). Yellow = present. "
           "[U] = Undergraduate, [G] = Graduate. Right column = years present.")}

<h2>3. Course Trajectories in the Didactic Space</h2>
<p>We embed all courses from all 5 years and track how they move in the shared embedding space.
Even courses with only names (2021–2023) occupy a semantic position that can be compared with
the same course's richer representation in 2024–2025.</p>
{fig_block("Multi-Year Course Trajectories (PCA)", "F2",
           "Each point is a course in a specific year. Arrows connect the same course "
           "across consecutive years. Labeled items are long-lived courses with significant drift.")}

<div class="insight">
    <h4>Trajectory Insight</h4>
    <p>Courses with the same name can drift substantially in embedding space when their
    description evolves. This "semantic drift" is invisible to traditional catalog analysis
    but measurable in the Didactic Space.</p>
</div>

{fig_block("Embedding Similarity Distribution (2024→2025)", "F7",
           "For courses present in both 2024 and 2025: distribution of cosine similarity "
           "between their embeddings. Higher = more stable content.")}

<h2>4. Objective Trajectories</h2>
<p>Each course is scored against ADS objectives (market/mission/university alignment).
Arrows trace how these scores evolved across years.</p>
{fig_block("Market × Mission Objective Trajectories", "F3",
           "Each arrow traces a course through objective space across years. "
           "Right = higher market alignment, Up = higher mission alignment.")}

{fig_block("Objective Profile by Year (Radar)", "F8",
           "Mean objective scores per year. Shows systematic shifts in curriculum orientation.")}

<div class="insight">
    <h4>Objective Shift Summary</h4>
    <p>{'<br>'.join(f"<b>{obj.capitalize()}</b>: 2024 mean = {v['mean_2024']:.3f} → 2025 mean = {v['mean_2025']:.3f} (Δ = {v['delta']:+.3f})" for obj, v in obj_shift.items())}</p>
</div>

<h2>5. Competency Evolution</h2>
{fig_block("Competency Distribution", "F4",
           "Left: competency type counts by year (УК/ОПК/ПК). "
           "Right: per-FGOS competency profile heatmap.")}

<h2>6. Credit Redistribution</h2>
{fig_block("Credit Changes", "F5",
           "Scatter: 2024 vs 2025 credits (diagonal = no change). "
           "Histogram: distribution of credit deltas.")}

<h2>7. Course Lifecycle</h2>
{fig_block("Course Lifecycle", "F6",
           "Left: 2024↔2025 lifecycle (continuous/new/retired). "
           "Right: course longevity distribution (years present across all data).")}

<h2>8. Long-Lived Courses (3+ years)</h2>
<p>Courses that persist across 3 or more years are the institutional backbone.
Their trajectory in embedding space reveals how the core curriculum evolves.</p>
<table>
    <tr><th>Course</th><th>FGOS</th><th>Years</th><th>Total Sim</th><th>Consecutive Sims</th></tr>
    {long_rows}
</table>

<h2>9. Top Drifters and Anchors</h2>
<p><strong>Drifters</strong>: courses whose embedding changed most (2024→2025).
<strong>Anchors</strong>: the most stable courses.</p>
<h3>Top 15 Drifters</h3>
<table>
    <tr><th>Course</th><th>FGOS</th><th>Years</th><th>Cos Sim</th><th>Δ Market</th><th>Δ Mission</th></tr>
    {fmt_items(drifters)}
</table>
<h3>Top 15 Anchors</h3>
<table>
    <tr><th>Course</th><th>FGOS</th><th>Years</th><th>Cos Sim</th><th>Δ Market</th><th>Δ Mission</th></tr>
    {fmt_items(anchors)}
</table>

<h2>10. Methodology</h2>
<div class="insight">
    <h4>What the Didactic Space Reveals</h4>
    <p>Traditional curriculum analysis compares <em>names</em> and <em>credits</em>.
    The Didactic Space adds semantic positioning: each course occupies a point in a 384-dimensional
    space, benchmarked against labor market demands, institutional mission, and peer curricula.
    Temporal tracking turns these points into <em>trajectories</em>, revealing:</p>
    <ul style="margin-left: 2rem; margin-top: 0.5rem;">
        <li><b>Invisible drift</b>: courses with the same name that shift content over time</li>
        <li><b>Structural evolution</b>: program-level changes in didactic composition</li>
        <li><b>Objective realignment</b>: curriculum moving toward or away from market/mission targets</li>
        <li><b>Competency redistribution</b>: how competency profiles adapt to new FGOS standards</li>
    </ul>
</div>

</div>
</body>
</html>"""

    path = OUT_DIR / "MISIS_EVOLUTION_REPORT.html"
    path.write_text(html, encoding='utf-8')
    print(f"  Saved report: {path}")
    return path


# ═══════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MISIS Curriculum Evolution Analysis (All 5 Years)")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──
    print("\n[1/8] Loading data...")
    catalog = load_catalog()
    all_courses = load_all_years()
    competencies = load_competencies()
    snapshots = load_temporal_snapshots()

    print(f"  Catalog entries: {len(catalog)}")
    print(f"  Competency mappings: {len(competencies)}")

    # ── Step 2: Build multi-year index ──
    print("\n[2/8] Building multi-year course index...")
    course_index = build_multi_year_index(all_courses)
    multi_year = {k: v for k, v in course_index.items() if len(v) >= 2}
    longevity_dist = Counter(len(v) for v in course_index.values())
    print(f"  Unique courses: {len(course_index)}")
    print(f"  Courses in 2+ years: {len(multi_year)}")
    print(f"  Longevity distribution: {dict(sorted(longevity_dist.items()))}")

    # Lifecycle stats for 2024↔2025
    keys_24 = {k for k, v in course_index.items() if 2024 in v}
    keys_25 = {k for k, v in course_index.items() if 2025 in v}
    matched_24_25 = keys_24 & keys_25
    only_24 = keys_24 - keys_25
    only_25 = keys_25 - keys_24
    print(f"  2024↔2025: matched={len(matched_24_25)}, only2024={len(only_24)}, only2025={len(only_25)}")

    # ── Step 3: Build texts and embed ──
    print("\n[3/8] Building course texts for all years...")
    # Build texts per year per course key
    all_texts = []
    all_keys = []
    all_years_list = []
    key_year_to_idx = {}

    for key, year_dict in course_index.items():
        for year, row in year_dict.items():
            text = build_course_text(row)
            if text.strip():
                idx = len(all_texts)
                all_texts.append(text)
                all_keys.append(key)
                all_years_list.append(year)
                key_year_to_idx[(key, year)] = idx

    print(f"  Total texts to embed: {len(all_texts)}")

    print("\n[4/8] Encoding all course texts...")
    model = load_encoder()
    all_vecs = encode_texts(model, all_texts, f"all {len(all_texts)} courses")

    # Build embeddings_by_year dict for trajectory analysis
    embeddings_by_year = defaultdict(dict)
    for i, (key, year) in enumerate(zip(all_keys, all_years_list)):
        embeddings_by_year[year][key] = all_vecs[i]

    # ── Step 4: Build objective targets and score ──
    print("\n[5/8] Building objective targets and scoring...")
    targets_data = load_target_artifacts()
    print(f"  JOB_ROLE: {len(targets_data['JOB_ROLE'])}, "
          f"MISSION: {len(targets_data['MISSION'])}, "
          f"COURSE: {len(targets_data['COURSE'])}")

    market_texts = [a['text'] for a in targets_data['JOB_ROLE']][:500]
    mission_texts = [a['text'] for a in targets_data['MISSION']]
    uni_texts = [a['text'] for a in targets_data['COURSE']
                 if a.get('metadata', {}).get('university') == 'MISIS'][:500]

    market_vecs = encode_texts(model, market_texts, "market targets")
    mission_vecs = encode_texts(model, mission_texts, "mission targets")
    uni_vecs = encode_texts(model, uni_texts, "university targets")

    centroids = {
        'market': build_centroid(market_vecs),
        'mission': build_centroid(mission_vecs),
        'university': build_centroid(uni_vecs),
    }
    obj_names = list(centroids.keys())

    # ── Step 5: Analyze trajectories ──
    print("\n[6/8] Analyzing multi-year trajectories...")
    trajectories = analyze_multi_year_trajectories(course_index, embeddings_by_year)
    print(f"  Trajectories (2+ years with embeddings): {len(trajectories)}")

    long_trajs = sorted([t for t in trajectories if len(t['years']) >= 3],
                        key=lambda t: -len(t['years']))
    print(f"  Long trajectories (3+ years): {len(long_trajs)}")

    # Competency & credit analysis
    comp_analysis = analyze_competency_evolution(competencies)
    credit_changes = analyze_credit_changes(course_index)
    landscape = analyze_program_landscape(catalog)

    # ── Step 6: Generate figures ──
    print("\n[7/8] Generating figures...")
    figures = {}
    figures['F1'] = fig_program_landscape(landscape)
    figures['F2'] = fig_multi_year_drift(trajectories)
    figures['F3'] = fig_objective_trajectories(trajectories, centroids, obj_names)
    figures['F4'] = fig_competency_evolution(comp_analysis)
    figures['F5'] = fig_credit_redistribution(credit_changes)

    lifecycle_stats = {
        'matched_24_25': len(matched_24_25),
        'only_2024': len(only_24),
        'only_2025': len(only_25),
        'longevity_dist': dict(longevity_dist),
    }
    figures['F6'] = fig_course_lifecycle(lifecycle_stats)

    f7_result = fig_similarity_distribution(trajectories)
    if f7_result:
        figures['F7'] = f7_result[0]
        sims_24_25 = f7_result[1]
    else:
        sims_24_25 = np.array([])

    figures['F8'] = fig_objective_radar(trajectories, centroids, obj_names)

    # Build drifters/anchors for 2024→2025 pairs
    drifters = []
    anchors = []
    pairs_24_25 = [(t, cosine_sim(t['embeddings'][2024], t['embeddings'][2025]))
                   for t in trajectories if 2024 in t['years'] and 2025 in t['years']]
    pairs_24_25.sort(key=lambda x: x[1])

    for t, sim in pairs_24_25[:15]:
        m24 = cosine_sim(t['embeddings'][2024], centroids['market'])
        m25 = cosine_sim(t['embeddings'][2025], centroids['market'])
        mi24 = cosine_sim(t['embeddings'][2024], centroids['mission'])
        mi25 = cosine_sim(t['embeddings'][2025], centroids['mission'])
        drifters.append({
            'name': t['name'], 'fgos': t['fgos'],
            'years_str': '→'.join(str(y) for y in t['years']),
            'cos_sim': sim, 'market_delta': m25 - m24, 'mission_delta': mi25 - mi24,
        })
    for t, sim in pairs_24_25[-15:][::-1]:
        m24 = cosine_sim(t['embeddings'][2024], centroids['market'])
        m25 = cosine_sim(t['embeddings'][2025], centroids['market'])
        mi24 = cosine_sim(t['embeddings'][2024], centroids['mission'])
        mi25 = cosine_sim(t['embeddings'][2025], centroids['mission'])
        anchors.append({
            'name': t['name'], 'fgos': t['fgos'],
            'years_str': '→'.join(str(y) for y in t['years']),
            'cos_sim': sim, 'market_delta': m25 - m24, 'mission_delta': mi25 - mi24,
        })

    # Objective shift summary
    obj_shift = {}
    for obj in obj_names:
        s24 = [cosine_sim(t['embeddings'][2024], centroids[obj])
               for t in trajectories if 2024 in t['years']]
        s25 = [cosine_sim(t['embeddings'][2025], centroids[obj])
               for t in trajectories if 2025 in t['years']]
        if s24 and s25:
            obj_shift[obj] = {
                'mean_2024': float(np.mean(s24)),
                'mean_2025': float(np.mean(s25)),
                'delta': float(np.mean(s25) - np.mean(s24)),
            }

    # ── Step 7: Generate HTML report ──
    print("\n[8/8] Generating HTML report...")
    stats = {
        'total_unique_courses': len(course_index),
        'multi_year_courses': len(multi_year),
        'total_programs': len(set(r['program_code'] for r in catalog)),
        'courses_per_year': {y: len(c) for y, c in all_courses.items()},
        'programs_per_year': landscape['per_year'],
        'mean_cos_sim_24_25': float(np.mean(sims_24_25)) if len(sims_24_25) > 0 else 0,
    }

    results = {
        'figures': figures,
        'stats': stats,
        'top_drifters': drifters,
        'top_anchors': anchors,
        'objective_shift': obj_shift,
        'long_trajectories': long_trajs,
    }
    generate_html_report(results)

    # Save summary JSON
    summary = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'stats': stats,
        'longevity_distribution': dict(longevity_dist),
        'program_landscape': landscape['per_year'],
        'objective_shift': obj_shift,
    }
    if len(sims_24_25) > 0:
        summary['cosine_similarity_24_25'] = {
            'mean': float(np.mean(sims_24_25)),
            'median': float(np.median(sims_24_25)),
            'std': float(np.std(sims_24_25)),
            'min': float(np.min(sims_24_25)),
            'max': float(np.max(sims_24_25)),
            'p10': float(np.percentile(sims_24_25, 10)),
            'p90': float(np.percentile(sims_24_25, 90)),
        }
    with open(OUT_DIR / "evolution_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Evolution analysis complete!")
    print(f"  Report:  {OUT_DIR / 'MISIS_EVOLUTION_REPORT.html'}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Summary: {OUT_DIR / 'evolution_summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
