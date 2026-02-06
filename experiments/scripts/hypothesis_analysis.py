#!/usr/bin/env python3
"""
Hypothesis Analysis for ADS Cross-National Curriculum Study
Tests 6 hypotheses about why international universities contribute to Pareto frontier.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

def load_data():
    """Load Pareto results and course metadata."""
    base = Path("/Users/aleksandrvolkov/Desktop/ADS")

    # Load Pareto results
    with open(base / "experiments/reports/unified_v4_international/pareto.json") as f:
        pareto_data = json.load(f)

    # Load artifacts for metadata
    artifacts = []
    artifacts_path = base / "data/processed/unified_v4/artifacts.jsonl"
    if artifacts_path.exists():
        with open(artifacts_path) as f:
            for line in f:
                if line.strip():
                    artifacts.append(json.loads(line))

    # Also try courses.csv for richer metadata
    courses_meta = {}
    courses_path = base / "data/processed/unified_v4/courses.csv"
    if courses_path.exists():
        import csv
        with open(courses_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                course_id = row.get('id') or row.get('course_id')
                if course_id:
                    courses_meta[course_id] = row

    return pareto_data, artifacts, courses_meta


def extract_department(text, university):
    """Extract department/discipline from course text."""
    # Common department patterns
    dept_patterns = [
        r'Department[:\s]+([^.]+)',
        r'School of\s+([^.]+)',
        r'College of\s+([^.]+)',
        r'Division of\s+([^.]+)',
    ]

    for pattern in dept_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:50]

    # Try to infer from course code patterns
    if 'CS ' in text or 'Computer Science' in text.lower():
        return 'Computer Science'
    if 'Math' in text or 'MATH' in text:
        return 'Mathematics'
    if 'Physics' in text or 'PHYS' in text:
        return 'Physics'
    if 'Engineering' in text.lower():
        return 'Engineering'
    if 'Business' in text or 'MBA' in text:
        return 'Business'
    if 'Economics' in text or 'ECON' in text:
        return 'Economics'
    if 'Biology' in text or 'BIO' in text:
        return 'Biology'
    if 'Chemistry' in text or 'CHEM' in text:
        return 'Chemistry'
    if 'Philosophy' in text or 'PHIL' in text:
        return 'Philosophy'
    if 'History' in text or 'HIST' in text:
        return 'History'
    if 'Political' in text or 'Government' in text:
        return 'Political Science'
    if 'Psychology' in text or 'PSYCH' in text:
        return 'Psychology'
    if 'Electrical' in text or 'EECS' in text or 'EE ' in text:
        return 'Electrical Engineering'
    if 'Mechanical' in text or 'MechE' in text:
        return 'Mechanical Engineering'
    if 'Data Science' in text.lower() or 'Machine Learning' in text:
        return 'Data Science/ML'
    if 'Medicine' in text or 'Medical' in text:
        return 'Medicine'
    if 'Law' in text or 'Legal' in text:
        return 'Law'

    return 'Unknown'


def classify_discipline(dept):
    """Classify department into broad discipline categories."""
    dept_lower = dept.lower()

    # STEM
    stem_keywords = ['computer', 'math', 'physics', 'engineering', 'chemistry',
                     'biology', 'data science', 'ml', 'electrical', 'mechanical',
                     'statistics', 'astronomy', 'geology', 'materials']
    for kw in stem_keywords:
        if kw in dept_lower:
            return 'STEM'

    # Business/Economics
    if any(kw in dept_lower for kw in ['business', 'economics', 'finance', 'mba', 'management']):
        return 'Business/Econ'

    # Social Sciences
    if any(kw in dept_lower for kw in ['psychology', 'sociology', 'political', 'anthropology']):
        return 'Social Sciences'

    # Humanities
    if any(kw in dept_lower for kw in ['history', 'philosophy', 'literature', 'language', 'art', 'music']):
        return 'Humanities'

    # Medicine/Health
    if any(kw in dept_lower for kw in ['medicine', 'medical', 'health', 'nursing']):
        return 'Medicine/Health'

    # Law
    if 'law' in dept_lower or 'legal' in dept_lower:
        return 'Law'

    return 'Other'


def extract_level(text):
    """Extract course level (intro/advanced/graduate)."""
    text_lower = text.lower()

    # Graduate indicators
    if any(kw in text_lower for kw in ['graduate', 'phd', 'doctoral', 'advanced seminar', 'level: g']):
        return 'Graduate'

    # Advanced undergrad
    if any(kw in text_lower for kw in ['advanced', 'senior', 'upper-division', '400-level', '300-level']):
        return 'Advanced'

    # Introductory
    if any(kw in text_lower for kw in ['introduction', 'introductory', 'fundamentals', 'principles of',
                                        'beginner', '100-level', '101', 'basics']):
        return 'Introductory'

    return 'Intermediate'


def hypothesis_1_discipline_clustering(pareto_options, artifacts_dict):
    """H1: Universal vs country-specific courses by discipline."""
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Universal vs Country-Specific Courses by Discipline")
    print("="*70)

    # Count disciplines by university
    uni_disciplines = defaultdict(lambda: defaultdict(int))
    pareto_disciplines = defaultdict(lambda: defaultdict(int))

    for opt in pareto_options:
        oid = opt['option_id']
        uni = oid.split(':')[1] if ':' in oid else 'unknown'

        # Get text from artifacts
        text = artifacts_dict.get(oid, {}).get('text', '')
        dept = extract_department(text, uni)
        discipline = classify_discipline(dept)

        pareto_disciplines[uni][discipline] += 1

    print("\nPareto courses by University and Discipline:")
    print("-" * 60)

    # Create table
    disciplines = ['STEM', 'Business/Econ', 'Social Sciences', 'Humanities', 'Medicine/Health', 'Law', 'Other']
    unis = ['MIT', 'UC Berkeley', 'Stanford', 'KTH', 'Cornell', 'UIUC', 'Edinburgh']

    # Header
    print(f"{'University':<15}", end='')
    for d in disciplines:
        print(f"{d[:10]:<12}", end='')
    print("Total")
    print("-" * 100)

    for uni in unis:
        print(f"{uni:<15}", end='')
        total = 0
        for d in disciplines:
            count = pareto_disciplines[uni].get(d, 0)
            total += count
            print(f"{count:<12}", end='')
        print(total)

    # Key insight
    print("\n>>> Key Finding:")
    stem_intl = pareto_disciplines['KTH'].get('STEM', 0) + pareto_disciplines['Edinburgh'].get('STEM', 0)
    stem_us = sum(pareto_disciplines[u].get('STEM', 0) for u in ['MIT', 'UC Berkeley', 'Stanford', 'Cornell', 'UIUC'])
    print(f"    STEM in Pareto: International={stem_intl}, US={stem_us}")

    return pareto_disciplines


def hypothesis_2_departmental_effect(pareto_options, artifacts_dict):
    """H2: Disciplinary effect - technical vs comprehensive universities."""
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Disciplinary Effect (Technical vs Comprehensive)")
    print("="*70)

    # KTH is a technical university - compare its discipline mix
    uni_dept_counts = defaultdict(lambda: defaultdict(int))

    for opt in pareto_options:
        oid = opt['option_id']
        uni = oid.split(':')[1] if ':' in oid else 'unknown'
        text = artifacts_dict.get(oid, {}).get('text', '')
        dept = extract_department(text, uni)
        uni_dept_counts[uni][dept] += 1

    print("\nTop 5 departments per university in Pareto:")
    print("-" * 60)

    for uni in ['KTH', 'MIT', 'UC Berkeley', 'Stanford']:
        depts = uni_dept_counts[uni]
        sorted_depts = sorted(depts.items(), key=lambda x: -x[1])[:5]
        print(f"\n{uni}:")
        for dept, count in sorted_depts:
            print(f"  {dept}: {count}")

    return uni_dept_counts


def hypothesis_3_course_level(pareto_options, artifacts_dict):
    """H3: Course level effect - intro courses more universal?"""
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Course Level Effect (Intro vs Advanced)")
    print("="*70)

    uni_levels = defaultdict(lambda: defaultdict(int))

    for opt in pareto_options:
        oid = opt['option_id']
        uni = oid.split(':')[1] if ':' in oid else 'unknown'
        text = artifacts_dict.get(oid, {}).get('text', '')
        level = extract_level(text)
        uni_levels[uni][level] += 1

    print("\nPareto courses by University and Level:")
    print("-" * 60)

    levels = ['Introductory', 'Intermediate', 'Advanced', 'Graduate']
    unis = ['MIT', 'UC Berkeley', 'Stanford', 'KTH', 'Cornell', 'UIUC', 'Edinburgh']

    print(f"{'University':<15}", end='')
    for lvl in levels:
        print(f"{lvl:<14}", end='')
    print()
    print("-" * 75)

    for uni in unis:
        print(f"{uni:<15}", end='')
        for lvl in levels:
            count = uni_levels[uni].get(lvl, 0)
            print(f"{count:<14}", end='')
        print()

    # Calculate ratios
    print("\n>>> Key Finding:")
    for uni in ['KTH', 'MIT']:
        total = sum(uni_levels[uni].values())
        grad = uni_levels[uni].get('Graduate', 0)
        intro = uni_levels[uni].get('Introductory', 0)
        if total > 0:
            print(f"    {uni}: {grad/total*100:.1f}% graduate, {intro/total*100:.1f}% introductory")

    return uni_levels


def hypothesis_4_market_mission_tradeoff(pareto_options):
    """H4: Market-Mission trade-off curve by university."""
    print("\n" + "="*70)
    print("HYPOTHESIS 4: Market-Mission Trade-off Curve")
    print("="*70)

    uni_scores = defaultdict(list)

    for opt in pareto_options:
        oid = opt['option_id']
        uni = oid.split(':')[1] if ':' in oid else 'unknown'
        market = opt['objectives']['market']
        mission = opt['objectives']['mission']
        uni_scores[uni].append((market, mission))

    print("\nMarket-Mission profile by University (Pareto courses only):")
    print("-" * 70)
    print(f"{'University':<15} {'Mean Market':<12} {'Mean Mission':<12} {'Correlation':<12} {'Range'}")
    print("-" * 70)

    for uni in ['MIT', 'UC Berkeley', 'Stanford', 'KTH', 'Cornell', 'UIUC', 'Edinburgh']:
        scores = uni_scores[uni]
        if len(scores) > 1:
            markets = [s[0] for s in scores]
            missions = [s[1] for s in scores]
            corr = np.corrcoef(markets, missions)[0, 1]
            print(f"{uni:<15} {np.mean(markets):.3f}        {np.mean(missions):.3f}        {corr:+.3f}        "
                  f"M:[{min(markets):.2f}-{max(markets):.2f}]")
        elif len(scores) == 1:
            print(f"{uni:<15} {scores[0][0]:.3f}        {scores[0][1]:.3f}        N/A          single point")

    print("\n>>> Key Finding:")
    kth_scores = uni_scores['KTH']
    mit_scores = uni_scores['MIT']
    print(f"    KTH occupies different region: market range [{min(s[0] for s in kth_scores):.2f}-{max(s[0] for s in kth_scores):.2f}]")
    print(f"    MIT occupies: market range [{min(s[0] for s in mit_scores):.2f}-{max(s[0] for s in mit_scores):.2f}]")

    return uni_scores


def hypothesis_5_boundary_objects(pareto_options, artifacts_dict):
    """H5: Boundary objects - courses high on BOTH market AND mission."""
    print("\n" + "="*70)
    print("HYPOTHESIS 5: Boundary Objects (High on Both Market AND Mission)")
    print("="*70)

    # Find courses in top quartile for BOTH objectives
    markets = [opt['objectives']['market'] for opt in pareto_options]
    missions = [opt['objectives']['mission'] for opt in pareto_options]

    market_q75 = np.percentile(markets, 75)
    mission_q75 = np.percentile(missions, 75)

    print(f"\nThresholds: Market Q75={market_q75:.3f}, Mission Q75={mission_q75:.3f}")

    boundary_objects = []
    for opt in pareto_options:
        if opt['objectives']['market'] >= market_q75 and opt['objectives']['mission'] >= mission_q75:
            oid = opt['option_id']
            uni = oid.split(':')[1] if ':' in oid else 'unknown'
            text = artifacts_dict.get(oid, {}).get('text', '')[:100]
            boundary_objects.append({
                'id': oid,
                'university': uni,
                'market': opt['objectives']['market'],
                'mission': opt['objectives']['mission'],
                'preview': text
            })

    print(f"\nFound {len(boundary_objects)} boundary objects (top 25% on BOTH dimensions):")
    print("-" * 70)

    # Count by university
    uni_counts = defaultdict(int)
    for bo in boundary_objects:
        uni_counts[bo['university']] += 1

    print("\nBoundary objects by university:")
    for uni, count in sorted(uni_counts.items(), key=lambda x: -x[1]):
        print(f"  {uni}: {count}")

    print("\nTop 5 boundary objects:")
    sorted_bo = sorted(boundary_objects, key=lambda x: x['market'] + x['mission'], reverse=True)[:5]
    for bo in sorted_bo:
        print(f"\n  {bo['university']}: market={bo['market']:.3f}, mission={bo['mission']:.3f}")
        print(f"    {bo['preview']}...")

    return boundary_objects


def hypothesis_6_complementary_niches(pareto_options):
    """H6: Complementary niches - hypervolume contribution by university."""
    print("\n" + "="*70)
    print("HYPOTHESIS 6: Complementary Niches (Hypervolume Contribution)")
    print("="*70)

    def compute_hypervolume_2d(points, ref_point=(0, 0)):
        """Simple 2D hypervolume for market-mission."""
        if not points:
            return 0.0
        # Sort by first objective descending
        sorted_points = sorted(points, key=lambda x: -x[0])
        hv = 0.0
        prev_y = ref_point[1]
        for x, y in sorted_points:
            if y > prev_y:
                hv += (x - ref_point[0]) * (y - prev_y)
                prev_y = y
        return hv

    # Get all points
    all_points = [(opt['objectives']['market'], opt['objectives']['mission']) for opt in pareto_options]
    full_hv = compute_hypervolume_2d(all_points)

    print(f"\nFull Pareto hypervolume (2D market-mission): {full_hv:.4f}")
    print("\nHypervolume contribution by university (leave-one-out):")
    print("-" * 60)

    unis = ['MIT', 'UC Berkeley', 'Stanford', 'KTH', 'Cornell', 'UIUC', 'Edinburgh']
    contributions = {}

    for uni in unis:
        # Points without this university
        other_points = [(opt['objectives']['market'], opt['objectives']['mission'])
                        for opt in pareto_options
                        if uni not in opt['option_id']]
        hv_without = compute_hypervolume_2d(other_points)
        contribution = full_hv - hv_without
        contribution_pct = (contribution / full_hv) * 100 if full_hv > 0 else 0
        contributions[uni] = (contribution, contribution_pct)

        n_courses = sum(1 for opt in pareto_options if uni in opt['option_id'])
        print(f"  {uni:<15}: HV contribution = {contribution:.4f} ({contribution_pct:5.1f}%), {n_courses} courses")

    print("\n>>> Key Finding:")
    sorted_contribs = sorted(contributions.items(), key=lambda x: -x[1][0])
    print(f"    Largest HV contributor: {sorted_contribs[0][0]} ({sorted_contribs[0][1][1]:.1f}%)")
    print(f"    KTH contributes {contributions['KTH'][1]:.1f}% of hypervolume despite lower market alignment")

    return contributions


def generate_latex_table(results):
    """Generate LaTeX table summarizing all hypotheses."""
    print("\n" + "="*70)
    print("LATEX TABLE: Hypothesis Test Summary")
    print("="*70)

    latex = r"""
\begin{table*}[t]
\centering
\caption{Cross-National Curriculum Analysis: Six Hypotheses Tested on unified\_v4 (7 universities, 3 countries)}
\label{tab:hypothesis-analysis}
\begin{tabular}{p{3cm}p{5cm}p{6cm}}
\toprule
\textbf{Hypothesis} & \textbf{Test} & \textbf{Finding} \\
\midrule
H1: Universal vs Country-Specific & Discipline clustering in Pareto & [STEM_FINDING] \\
H2: Disciplinary Effect & Technical vs comprehensive universities & [DEPT_FINDING] \\
H3: Course Level & Intro vs advanced in Pareto & [LEVEL_FINDING] \\
H4: Market-Mission Trade-off & Score profiles by university & [TRADEOFF_FINDING] \\
H5: Boundary Objects & Courses high on both dimensions & [BOUNDARY_FINDING] \\
H6: Complementary Niches & Hypervolume contribution & [HV_FINDING] \\
\bottomrule
\end{tabular}
\end{table*}
"""
    print(latex)


def main():
    print("="*70)
    print("ADS HYPOTHESIS ANALYSIS: Cross-National Curriculum Study")
    print("="*70)

    # Load data
    print("\nLoading data...")
    pareto_data, artifacts, courses_meta = load_data()
    pareto_options = pareto_data.get('pareto', [])

    print(f"Loaded {len(pareto_options)} Pareto options")
    print(f"Loaded {len(artifacts)} artifacts")

    # Build artifacts lookup
    artifacts_dict = {}
    for art in artifacts:
        aid = art.get('id', '')
        artifacts_dict[aid] = art

    # Run all hypothesis tests
    h1_results = hypothesis_1_discipline_clustering(pareto_options, artifacts_dict)
    h2_results = hypothesis_2_departmental_effect(pareto_options, artifacts_dict)
    h3_results = hypothesis_3_course_level(pareto_options, artifacts_dict)
    h4_results = hypothesis_4_market_mission_tradeoff(pareto_options)
    h5_results = hypothesis_5_boundary_objects(pareto_options, artifacts_dict)
    h6_results = hypothesis_6_complementary_niches(pareto_options)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)

    print("""
1. DISCIPLINE CLUSTERING: [See H1 results above]
   - Check if STEM dominates international Pareto contributions

2. DISCIPLINARY EFFECT: [See H2 results above]
   - KTH as technical university has different department mix

3. COURSE LEVEL: [See H3 results above]
   - Compare graduate vs introductory ratios

4. MARKET-MISSION TRADE-OFF: [See H4 results above]
   - Universities occupy different regions of objective space

5. BOUNDARY OBJECTS: [See H5 results above]
   - Courses that bridge stakeholder needs

6. COMPLEMENTARY NICHES: [See H6 results above]
   - KTH contributes unique hypervolume despite lower alignment
""")

    # Save results
    results = {
        'h1_disciplines': {k: dict(v) for k, v in h1_results.items()},
        'h2_departments': {k: dict(v) for k, v in h2_results.items()},
        'h3_levels': {k: dict(v) for k, v in h3_results.items()},
        'h4_scores': {k: [(m, mi) for m, mi in v] for k, v in h4_results.items()},
        'h5_boundary_objects': h5_results,
        'h6_hv_contributions': {k: {'contribution': v[0], 'percentage': v[1]} for k, v in h6_results.items()}
    }

    output_path = Path("/Users/aleksandrvolkov/Desktop/ADS/experiments/reports/hypothesis_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
