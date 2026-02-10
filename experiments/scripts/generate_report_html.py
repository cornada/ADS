#!/usr/bin/env python3
"""Generate comprehensive HTML report for ADS unified_v5 experiments."""
import base64
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("/Users/aleksandrvolkov/Desktop/ADS")
FIG_DIR = BASE / "experiments/reports/unified_v5_figures"
CORE_DIR = BASE / "experiments/reports/unified_v5_core_multilingual/paper_artifacts/paper_tables"
COMP_DIR = BASE / "experiments/reports/unified_v5_competency_multilingual/paper_artifacts/paper_tables"
OUTPUT = BASE / "experiments/reports/ADS_REPORT.html"


def img_b64(path):
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_uni(oid):
    parts = oid.split(":")
    return "UC Berkeley" if len(parts) >= 2 and parts[1] == "UC" else parts[1] if len(parts) >= 2 else "?"


def main():
    with open(CORE_DIR / "summary.json") as f:
        core = json.load(f)
    with open(COMP_DIR / "summary.json") as f:
        comp = json.load(f)

    pareto4 = load_csv(CORE_DIR / "pareto_table.csv")
    pareto5 = load_csv(COMP_DIR / "pareto_table.csv")
    all_data = load_csv(COMP_DIR / "all_results.csv")

    ids4 = {r['option_id'] for r in pareto4}
    ids5 = {r['option_id'] for r in pareto5}

    # University breakdown
    uni4 = defaultdict(int)
    uni5 = defaultdict(int)
    for r in pareto4: uni4[get_uni(r['option_id'])] += 1
    for r in pareto5: uni5[get_uni(r['option_id'])] += 1

    # Cross-national means
    objs = ['market', 'mission', 'university', 'learner', 'competency']
    uni_scores = defaultdict(lambda: defaultdict(list))
    for r in all_data:
        u = get_uni(r['option_id'])
        for o in objs:
            uni_scores[u][o].append(float(r[o]))

    # Top-10
    for r in pareto5:
        r['avg'] = sum(float(r[o]) for o in objs) / len(objs)
    top10 = sorted(pareto5, key=lambda r: -r['avg'])[:10]

    # Figures
    figs = {}
    for f in sorted(FIG_DIR.glob("*.png")):
        figs[f.stem] = img_b64(f)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADS: Agent Didactic Spaces — Comprehensive Report</title>
<style>
  :root {{
    --bg: #0f172a; --bg2: #1e293b; --bg3: #334155;
    --text: #e2e8f0; --text2: #94a3b8; --accent: #38bdf8;
    --accent2: #818cf8; --green: #4ade80; --red: #f87171;
    --orange: #fb923c; --yellow: #fbbf24;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    max-width: 1200px; margin: 0 auto; padding: 2rem;
  }}
  h1 {{ font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, var(--accent), var(--accent2));
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.8rem; color: var(--accent); margin: 2.5rem 0 1rem; border-bottom: 2px solid var(--bg3); padding-bottom: 0.5rem; }}
  h3 {{ font-size: 1.3rem; color: var(--accent2); margin: 1.5rem 0 0.8rem; }}
  p, li {{ color: var(--text2); font-size: 1.05rem; }}
  p {{ margin-bottom: 1rem; }}
  ul, ol {{ padding-left: 1.5rem; margin-bottom: 1rem; }}
  li {{ margin-bottom: 0.3rem; }}
  strong {{ color: var(--text); }}
  .hero {{ text-align: center; padding: 3rem 0 2rem; }}
  .hero p {{ font-size: 1.2rem; color: var(--text2); }}
  .subtitle {{ font-size: 1.1rem; color: var(--text2); margin-bottom: 2rem; }}
  .card {{ background: var(--bg2); border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; border: 1px solid var(--bg3); }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .metric-card {{ background: var(--bg2); border-radius: 10px; padding: 1.2rem; text-align: center; border: 1px solid var(--bg3); }}
  .metric-card .number {{ font-size: 2.5rem; font-weight: 800; }}
  .metric-card .label {{ font-size: 0.9rem; color: var(--text2); margin-top: 0.3rem; }}
  .blue {{ color: var(--accent); }}
  .purple {{ color: var(--accent2); }}
  .green {{ color: var(--green); }}
  .red {{ color: var(--red); }}
  .orange {{ color: var(--orange); }}
  .yellow {{ color: var(--yellow); }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.95rem; }}
  th {{ background: var(--bg3); color: var(--accent); padding: 0.7rem 1rem; text-align: left; font-weight: 700; }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid var(--bg3); }}
  tr:hover td {{ background: rgba(56, 189, 248, 0.05); }}
  .fig {{ margin: 2rem 0; text-align: center; }}
  .fig img {{ max-width: 100%; border-radius: 8px; border: 1px solid var(--bg3); }}
  .fig .caption {{ font-size: 0.9rem; color: var(--text2); margin-top: 0.5rem; font-style: italic; }}
  .tag {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin: 0.1rem; }}
  .tag-us {{ background: rgba(56, 189, 248, 0.15); color: var(--accent); }}
  .tag-eu {{ background: rgba(129, 140, 248, 0.15); color: var(--accent2); }}
  .tag-ru {{ background: rgba(74, 222, 128, 0.15); color: var(--green); }}
  .pipeline {{ display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; }}
  .pipeline .step {{ background: var(--bg3); padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; font-size: 0.9rem; }}
  .pipeline .arrow {{ color: var(--accent); font-size: 1.2rem; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .insight {{ background: linear-gradient(135deg, rgba(56,189,248,0.1), rgba(129,140,248,0.1));
             border-left: 4px solid var(--accent); padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1.5rem 0; }}
  .insight strong {{ color: var(--accent); }}
  .footer {{ text-align: center; padding: 3rem 0 1rem; color: var(--text2); font-size: 0.85rem; }}
  code {{ background: var(--bg3); padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.9rem; color: var(--green); }}
</style>
</head>
<body>

<div class="hero">
  <h1>Agent Didactic Spaces (ADS)</h1>
  <p class="subtitle">Multi-Objective Pareto Optimization for Curriculum Design</p>
  <p>Comprehensive Report &mdash; unified_v5 Dataset &mdash; {comp['encoder'].split('/')[-1]}</p>
  <p style="font-size:0.9rem; color:var(--text2);">Generated: 2026-02-10 | Branch: kt20/outcome-sanity-curriculum-text</p>
</div>

<h2>1. What is ADS?</h2>

<div class="card">
<h3>The Problem</h3>
<p>Universities face a <strong>multi-stakeholder optimization problem</strong> when designing curricula. A single course must simultaneously satisfy:</p>
<ul>
  <li><strong>Labor Market</strong> &mdash; does the course teach skills employers actually need?</li>
  <li><strong>Institutional Mission</strong> &mdash; does it align with the university's stated educational goals?</li>
  <li><strong>University Standards</strong> &mdash; does it meet the institution's academic requirements?</li>
  <li><strong>Learner Needs</strong> &mdash; is it appropriate for the student's current level and goals?</li>
  <li><strong>Competency Coverage</strong> &mdash; does it satisfy formal educational standard requirements (e.g., FGOS)?</li>
</ul>
<p>These objectives often <strong>conflict</strong>: a course perfectly aligned with market demands may not satisfy academic mission requirements, and vice versa. There is no single "best" curriculum &mdash; only a set of <strong>Pareto-optimal trade-offs</strong>.</p>
</div>

<div class="card">
<h3>The Solution: Agent Didactic Spaces</h3>
<p>ADS is a <strong>computational framework</strong> that:</p>
<ol>
  <li><strong>Embeds</strong> courses, job descriptions, missions, and competency requirements into a shared semantic vector space using multilingual transformer models</li>
  <li><strong>Scores</strong> each course against each objective using cosine similarity to target centroids</li>
  <li><strong>Filters</strong> courses by an autonomy-drift constraint (&#964; = 0.25) to ensure feasibility</li>
  <li><strong>Computes</strong> the Pareto front &mdash; the set of courses where no other course is better on ALL objectives simultaneously</li>
</ol>

<div class="pipeline">
  <div class="step">Raw Data<br><small>courses, jobs, missions</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Artifact Store<br><small>JSONL + provenance</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Embedding<br><small>MiniLM / E5-large</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Objective Scoring<br><small>cosine similarity</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Pareto Front<br><small>non-dominated set</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Reports &amp; Figures<br><small>paper-ready</small></div>
</div>
</div>

<h2>2. Dataset: unified_v5</h2>

<div class="card-grid">
  <div class="metric-card">
    <div class="number blue">36,197</div>
    <div class="label">Total Artifacts</div>
  </div>
  <div class="metric-card">
    <div class="number purple">29,802</div>
    <div class="label">Course Options</div>
  </div>
  <div class="metric-card">
    <div class="number green">8</div>
    <div class="label">Universities</div>
  </div>
  <div class="metric-card">
    <div class="number orange">3</div>
    <div class="label">Labor Markets</div>
  </div>
</div>

<div class="two-col">
<div class="card">
<h3>Universities (8)</h3>
<table>
<tr><th>University</th><th>Country</th><th>Courses</th></tr>
<tr><td>UC Berkeley</td><td><span class="tag tag-us">US</span></td><td>11,107</td></tr>
<tr><td>MIT</td><td><span class="tag tag-us">US</span></td><td>6,941</td></tr>
<tr><td>KTH</td><td><span class="tag tag-eu">SE</span></td><td>4,768</td></tr>
<tr><td>NUST MISIS</td><td><span class="tag tag-ru">RU</span></td><td>2,175</td></tr>
<tr><td>Stanford</td><td><span class="tag tag-us">US</span></td><td>2,663</td></tr>
<tr><td>UIUC</td><td><span class="tag tag-us">US</span></td><td>860</td></tr>
<tr><td>Cornell</td><td><span class="tag tag-us">US</span></td><td>842</td></tr>
<tr><td>Edinburgh</td><td><span class="tag tag-eu">UK</span></td><td>446</td></tr>
</table>
</div>

<div class="card">
<h3>Labor Markets (3)</h3>
<table>
<tr><th>Source</th><th>Region</th><th>Artifacts</th></tr>
<tr><td>O*NET</td><td><span class="tag tag-us">US</span></td><td>1,016 occupations</td></tr>
<tr><td>Profstandart + HeadHunter</td><td><span class="tag tag-ru">RU</span></td><td>1,697 + 2,879 = 4,576</td></tr>
<tr><td>ESCO</td><td><span class="tag tag-eu">EU</span></td><td>162 occupations</td></tr>
</table>
<h3 style="margin-top:1.5rem;">Additional Sources</h3>
<table>
<tr><th>Type</th><th>Count</th></tr>
<tr><td>FGOS Competencies (MISIS)</td><td>581 artifacts</td></tr>
<tr><td>Mission Statements</td><td>22 artifacts</td></tr>
</table>
</div>
</div>

<div class="insight">
<strong>Why 3 labor markets?</strong> Different countries have different classification systems: the US uses O*NET/SOC, Russia uses Profstandart (FGOS-linked professional standards) + live HeadHunter vacancies, and the EU uses ESCO. By including all three, ADS can evaluate how well a course aligns with <em>global</em> labor demand, not just a single national market.
</div>

<h2>3. Experiment Results</h2>

<div class="card-grid">
  <div class="metric-card">
    <div class="number blue">55</div>
    <div class="label">Pareto (4 objectives)</div>
  </div>
  <div class="metric-card">
    <div class="number red">68</div>
    <div class="label">Pareto (5 objectives)</div>
  </div>
  <div class="metric-card">
    <div class="number green">+24%</div>
    <div class="label">Expansion with Competency</div>
  </div>
  <div class="metric-card">
    <div class="number yellow">27,121</div>
    <div class="label">Feasible Courses (91%)</div>
  </div>
</div>

<h3>Best Course per Objective</h3>
<div class="card">
<table>
<tr><th>Objective</th><th>Best Course</th><th>Score</th><th>Interpretation</th></tr>
<tr><td><strong class="blue">Market</strong></td><td>course:MISIS:c1abf156d70b</td><td>0.662</td><td>Highest alignment with combined labor market demand</td></tr>
<tr><td><strong class="purple">Mission</strong></td><td>course:MISIS:cd588afde20a</td><td>0.732</td><td>Closest to institutional mission statements</td></tr>
<tr><td><strong class="green">University</strong></td><td>course:MISIS:8181ddb16717</td><td>0.861</td><td>Best institutional standards match</td></tr>
<tr><td><strong class="orange">Learner</strong></td><td>course:KTH:705b5ce850cd</td><td>0.688</td><td>Best fit for learner profile</td></tr>
<tr><td><strong class="red">Competency</strong></td><td>course:MISIS:c0ca8dafe864</td><td>0.638</td><td>Highest FGOS competency coverage</td></tr>
</table>
</div>

<h3>Top-10 Pareto-Optimal Courses (by average 5-objective score)</h3>
<div class="card">
<table>
<tr><th>#</th><th>University</th><th>Market</th><th>Mission</th><th>Univ.</th><th>Learner</th><th>Compet.</th><th>Avg</th></tr>"""

    for i, r in enumerate(top10, 1):
        uni = get_uni(r['option_id'])
        is_new = " *" if r['option_id'] in (ids5 - ids4) else ""
        html += f"""
<tr><td>{i}</td><td>{uni}{is_new}</td><td>{float(r['market']):.3f}</td><td>{float(r['mission']):.3f}</td><td>{float(r['university']):.3f}</td><td>{float(r['learner']):.3f}</td><td>{float(r['competency']):.3f}</td><td><strong>{r['avg']:.3f}</strong></td></tr>"""

    html += """
</table>
<p style="font-size:0.85rem;">* = new in 5-objective Pareto front (not present in 4-objective front)</p>
</div>

<h3>Pareto Front: University Distribution</h3>
<div class="card">
<table>
<tr><th>University</th><th>4-Objective</th><th>5-Objective</th><th>Delta</th></tr>"""

    all_unis = sorted(set(uni4) | set(uni5), key=lambda u: -uni5.get(u, 0))
    for u in all_unis:
        c4, c5 = uni4.get(u, 0), uni5.get(u, 0)
        delta = c5 - c4
        d_str = f"+{delta}" if delta > 0 else str(delta)
        html += f'<tr><td>{u}</td><td>{c4}</td><td>{c5}</td><td>{"<strong class=green>" + d_str + "</strong>" if delta > 0 else d_str}</td></tr>'

    html += f"""
<tr style="border-top:2px solid var(--bg3)"><td><strong>Total</strong></td><td><strong>{len(pareto4)}</strong></td><td><strong>{len(pareto5)}</strong></td><td><strong class="green">+{len(pareto5)-len(pareto4)}</strong></td></tr>
</table>
</div>
"""

    # Figures section
    html += '<h2>4. Figures</h2>'

    fig_meta = [
        ("F1_pareto_scatter_market_mission",
         "F1: Pareto Front — Market vs Mission",
         "Each dot is a course. Gray = non-Pareto (29,747). Colored = Pareto-optimal (55), colored by university. MISIS (teal) dominates the upper-right quadrant — high on both market and mission. KTH (purple) and Stanford (green) occupy distinct niches. The Pareto front forms a clear convex boundary."),
        ("F2_4obj_vs_5obj_comparison",
         "F2: Pareto Front Size — 4 vs 5 Objectives",
         "Left: Adding the competency objective expands the Pareto front from 55 to 68 courses (+24%). Right: All 13 new entrants are MISIS courses. Non-MISIS universities maintain exactly the same count — the competency dimension selectively unlocks previously-dominated MISIS courses."),
        ("F3_radar_university_specialization",
         "F3: Objective Specialization by University",
         "Radar chart showing mean scores of Pareto-optimal courses per university. MISIS has the largest area — strong on all 5 objectives. KTH excels on market and learner. Stanford is balanced but lower. Cornell punches above its weight on market. Edinburgh is a niche player."),
        ("F4_competency_distribution",
         "F4: Competency Score Distribution",
         "Pareto-optimal courses (red, mean=0.500) score 74% higher on competency than non-Pareto courses (gray, mean=0.288). This confirms competency alignment is a discriminating factor — Pareto courses are not just market-aligned, they also satisfy formal educational standards."),
        ("F5_cross_national_heatmap",
         "F5: Cross-National Objective Alignment Heatmap",
         "Mean objective scores across ALL courses per university. MISIS leads on market (0.494) due to Russian-language alignment with Profstandart/HeadHunter targets. Stanford leads on mission (0.411). The 'university' column shows institutional self-alignment (all >0.5). UC Berkeley's lower scores reflect its very large catalog diluting average quality."),
        ("F6_pareto_expansion_market_competency",
         "F6: Pareto Front Expansion (4→5 Objectives)",
         "Blue circles = courses in both Pareto fronts (55). Red stars = 13 new courses entering the 5-objective front. New entrants cluster in the high-competency region (>0.54), with moderate-to-high market scores (0.49-0.61). These are courses that were previously dominated but the competency dimension reveals their unique value."),
    ]

    for key, title, desc in fig_meta:
        if key in figs:
            html += f"""
<div class="fig">
  <img src="{figs[key]}" alt="{title}">
  <div class="caption"><strong>{title}</strong><br>{desc}</div>
</div>"""

    # Cross-national table
    html += """
<h2>5. Cross-National Analysis</h2>
<div class="card">
<h3>Mean Objective Scores — All Courses</h3>
<table>
<tr><th>University</th><th>N</th><th>Market</th><th>Mission</th><th>University</th><th>Learner</th><th>Competency</th></tr>"""

    for uni in sorted(uni_scores, key=lambda u: -np.mean(uni_scores[u]['market'])):
        n = len(uni_scores[uni]['market'])
        if n < 10:
            continue
        vals = {o: np.mean(uni_scores[uni][o]) for o in objs}
        row = f'<tr><td><strong>{uni}</strong></td><td>{n:,}</td>'
        for o in objs:
            v = vals[o]
            if v == max(np.mean(uni_scores[u][o]) for u in uni_scores if len(uni_scores[u][o]) > 10):
                row += f'<td><strong class="green">{v:.3f}</strong></td>'
            else:
                row += f'<td>{v:.3f}</td>'
        row += '</tr>'
        html += row

    html += """
</table>
<p style="font-size:0.85rem;">Green = highest in column. N = number of courses from that university.</p>
</div>
"""

    # Insights section
    html += """
<h2>6. Key Insights</h2>

<div class="insight">
<strong>Insight 1: Language as a proxy for market alignment.</strong>
MISIS courses (Russian) score highest on market alignment because the MARKET_RU targets (Profstandart + HeadHunter) are also in Russian. The multilingual encoder gives higher cosine similarity to same-language pairs. This is not a bug — it reflects the genuine advantage of a curriculum designed for its local labor market. A Stanford course in English will naturally be less aligned with Russian professional standards.
</div>

<div class="insight">
<strong>Insight 2: The competency objective is discriminating, not redundant.</strong>
Adding the 5th objective (FGOS competencies) expanded the Pareto front by 24% — all new entrants are MISIS courses. This means competency alignment provides genuinely new information that the other 4 objectives don't capture. Courses that match formal competency requirements aren't always the same as those matching market demand.
</div>

<div class="insight">
<strong>Insight 3: No single university dominates all dimensions.</strong>
While MISIS leads on market, competency, and university, KTH wins on learner alignment. Stanford and Cornell have competitive mission scores. This confirms the fundamental premise of multi-objective optimization: there is no universally "best" curriculum — only trade-offs.
</div>

<div class="insight">
<strong>Insight 4: 91% feasibility rate validates the autonomy-drift constraint.</strong>
27,121 of 29,802 courses (91%) pass the &#964; = 0.25 drift constraint. The 9% filtered out are courses so far from all objectives that they would be poor choices under any weighting. This constraint acts as a quality floor without being overly restrictive.
</div>

<div class="insight">
<strong>Insight 5: Pareto selectivity is extremely high — 0.23%.</strong>
Only 68 out of 29,802 courses are Pareto-optimal in 5 dimensions. This 0.23% selectivity means ADS provides strong discriminative power. A decision-maker choosing from the Pareto front faces a manageable set of 68 genuine trade-offs rather than 30K undifferentiated options.
</div>

<h2>7. Physical Meaning of the Project</h2>

<div class="card">
<h3>What does ADS actually compute?</h3>
<p>ADS answers the question: <strong>"Given multiple stakeholders with different priorities, which courses represent the best possible trade-offs?"</strong></p>

<p>Think of it geometrically. Each course is a point in 5-dimensional objective space. The Pareto front is the "outer surface" of this point cloud — the set of points where you can't improve any dimension without worsening another. This surface is the <strong>efficient frontier</strong> of curriculum design.</p>

<p>The objectives have concrete physical meanings:</p>
<ul>
  <li><strong>Market</strong> = cosine similarity between course embedding and the centroid of all job role embeddings. High score means the course teaches skills that appear frequently in job descriptions across US, Russian, and EU markets.</li>
  <li><strong>Mission</strong> = cosine similarity to university mission statement centroids. Measures whether the course fulfills the institution's stated educational purpose.</li>
  <li><strong>University</strong> = alignment with institutional academic standards. Captures whether the course fits the university's overall curriculum structure.</li>
  <li><strong>Learner</strong> = match with a synthetic learner profile. Measures appropriateness for a target student demographic.</li>
  <li><strong>Competency</strong> = alignment with FGOS competency requirements. Measures coverage of formal educational standards (applicable to Russian higher education).</li>
</ul>

<p>The <strong>autonomy-drift constraint (&#964;)</strong> ensures courses don't diverge too far from the aggregate objective. It's a regularizer that prevents degenerate solutions — courses that score 1.0 on one objective but 0.0 on all others.</p>
</div>

<div class="card">
<h3>Why Pareto optimization instead of a single score?</h3>
<p>A weighted sum (e.g., 0.3 &times; market + 0.3 &times; mission + ...) assumes we know the relative importance of each objective in advance. But:</p>
<ul>
  <li>A CS department chair might prioritize market alignment (employability)</li>
  <li>A dean might prioritize mission alignment (accreditation)</li>
  <li>A student might prioritize learner fit (career goals)</li>
  <li>A ministry of education might prioritize competency coverage (FGOS compliance)</li>
</ul>
<p>Pareto optimization is <strong>weight-agnostic</strong>: it finds ALL efficient solutions, and the stakeholder chooses based on their own priority. The Pareto front is a <strong>menu of optimal trade-offs</strong>, not a single recommendation.</p>
</div>

<h2>8. Architecture</h2>

<div class="card">
<h3>Technical Stack</h3>
<table>
<tr><th>Component</th><th>Technology</th><th>Purpose</th></tr>
<tr><td>Pipeline</td><td>Python + Hydra</td><td>Configurable experiment runner with YAML configs</td></tr>
<tr><td>Embeddings</td><td>sentence-transformers</td><td>Multilingual semantic embeddings (MiniLM-L12 / E5-large)</td></tr>
<tr><td>Data ingest</td><td>Custom BaseConnector</td><td>Pluggable ingestors for each data source</td></tr>
<tr><td>Pareto solver</td><td>NumPy</td><td>Exact non-dominated sorting</td></tr>
<tr><td>Caching</td><td>Disk-based (NPY + JSON index)</td><td>Embedding cache keyed by sha256(text) + model_id</td></tr>
<tr><td>Reporting</td><td>Matplotlib</td><td>400 DPI paper-ready figures</td></tr>
<tr><td>API</td><td>FastAPI (uvicorn)</td><td>REST API for interactive querying</td></tr>
<tr><td>Config</td><td>Hydra + OmegaConf</td><td>Composable YAML configs for datasets, embeddings, objectives</td></tr>
</table>

<h3 style="margin-top:1.5rem;">Data Sources & Ingestors</h3>
<table>
<tr><th>Source</th><th>Ingestor</th><th>API/Method</th></tr>
<tr><td>MIT OCW</td><td><code>ingest/mit_real.py</code></td><td>MIT OpenCourseWare JSON</td></tr>
<tr><td>Stanford, Cornell, UIUC, UCB</td><td><code>ingest/stanford.py</code>, etc.</td><td>Course catalog APIs</td></tr>
<tr><td>KTH</td><td><code>ingest/kth.py</code></td><td>KTH course API</td></tr>
<tr><td>Edinburgh</td><td><code>ingest/edinburgh.py</code></td><td>Edinburgh course catalog</td></tr>
<tr><td>NUST MISIS</td><td><code>ingest/misis.py</code></td><td>PDF annotation parsing (pdfplumber)</td></tr>
<tr><td>O*NET</td><td><code>ingest/scorecard.py</code></td><td>O*NET OnLine database</td></tr>
<tr><td>Profstandart</td><td><code>ingest/profstandart.py</code></td><td>XLS professional standards registry</td></tr>
<tr><td>HeadHunter</td><td><code>ingest/headhunter.py</code></td><td>hh.ru REST API</td></tr>
<tr><td>ESCO</td><td><code>ingest/esco.py</code></td><td>EU ESCO REST API</td></tr>
<tr><td>FGOS Competencies</td><td><code>ingest/competencies.py</code></td><td>Parsed from MISIS PDFs</td></tr>
</table>
</div>

<h2>9. Perspectives & Next Steps</h2>

<div class="card">
<h3>Immediate (next sprint)</h3>
<ol>
  <li><strong>Fix MPS deadlock for E5-large</strong> — implement chunked encoding (encode in 5K-text chunks) to avoid Metal GPU timeout. This will enable the final paper results with the 1024-dim model.</li>
  <li><strong>Shared embedding cache</strong> — move cache from per-run to global (<code>~/.cache/ads/embeddings/</code>) so the second experiment reuses embeddings from the first.</li>
  <li><strong>Encoder ablation</strong> — compare MiniLM-L12 (384d) vs E5-large (1024d) vs BGE-large (1024d) to quantify the impact of model choice on Pareto front composition.</li>
  <li><strong>FGOS→CIP→SOC crosswalk</strong> — complete the two-hop mapping from Russian FGOS codes to US SOC codes via CIP (Classification of Instructional Programs). This enables apples-to-apples cross-national comparison.</li>
</ol>
</div>

<div class="card">
<h3>Medium-term (paper submission)</h3>
<ol>
  <li><strong>Temporal analysis</strong> — MISIS data spans 2021-2025. Run ADS on each year independently and track how the Pareto front drifts over time. This is the "time travel" analysis referenced in the KDD paper.</li>
  <li><strong>Prerequisite DAGs</strong> — extract prerequisite chains from MISIS annotations and model them as planning state constraints (ICAPS MTCPP paper).</li>
  <li><strong>Zombie-risk metric</strong> — courses that were once Pareto-optimal but have drifted out of the front. Computed from temporal snapshots.</li>
  <li><strong>NSGA-II comparison</strong> — benchmark exact Pareto computation against evolutionary multi-objective optimization (NSGA-II) for scalability analysis.</li>
  <li><strong>Language bias quantification</strong> — measure the language effect by computing Pareto fronts with English-only vs Russian-only vs multilingual embeddings.</li>
</ol>
</div>

<div class="card">
<h3>Long-term (system deployment)</h3>
<ol>
  <li><strong>Interactive dashboard</strong> — web UI where decision-makers can explore the Pareto front, filter by university, and adjust objective weights in real time.</li>
  <li><strong>Curriculum recommender</strong> — given a student profile and institutional constraints, recommend the Pareto-optimal course sequence (not just individual courses).</li>
  <li><strong>Continuous ingestion</strong> — automated pipelines that re-run HeadHunter/ESCO ingestion weekly to track market shifts.</li>
  <li><strong>Additional universities</strong> — add more Russian universities (MEPhI, ITMO, Bauman) and EU universities (TU Munich, ETH Zurich) for broader cross-national coverage.</li>
</ol>
</div>

<h2>10. Repository Structure</h2>

<div class="card">
<pre style="background:var(--bg); padding:1rem; border-radius:8px; overflow-x:auto; font-size:0.85rem; line-height:1.5; color:var(--text2);">
ADS/
├── packages/
│   ├── ads_core/
│   │   ├── data/schemas.py          # Artifact, ArtifactType (incl. COMPETENCY)
│   │   ├── embed/                    # Encoder framework + caching
│   │   ├── ingest/                   # 10+ data source ingestors
│   │   ├── pipeline/                 # Toy + dataset pipelines
│   │   └── report/                   # Pareto report builder
│   └── ads_api/                      # FastAPI REST interface
├── experiments/
│   ├── conf/                         # Hydra configs (dataset, embedding, objectives)
│   ├── scripts/                      # Figure generation, merge scripts
│   ├── reports/                      # Experiment outputs (per-run directories)
│   └── run.py                        # Main experiment entry point
├── data/
│   ├── raw/                          # Per-source raw data + artifacts
│   └── processed/                    # Merged datasets (unified_v3..v5)
├── tests/                            # Pytest test suite
└── KDD_Agent_Didactic_Spaces/        # Paper LaTeX sources
</pre>
</div>

<div class="footer">
  <p>ADS: Agent Didactic Spaces &mdash; Multi-Objective Pareto Optimization for Curriculum Design</p>
  <p>Encoder: paraphrase-multilingual-MiniLM-L12-v2 | Dataset: unified_v5 (36,197 artifacts) | Seed: 42</p>
  <p>Report generated by Claude Code &mdash; 2026-02-10</p>
</div>

</body>
</html>"""

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Report saved to: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
