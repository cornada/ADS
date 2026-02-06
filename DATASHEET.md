# Datasheet for ADS-Unified-v4 Dataset

**Dataset Name:** ADS-Unified-v4
**Version:** 4.0.0
**Date:** February 2026
**DOI:** [To be assigned upon Zenodo upload]

---

## 1. Motivation

### 1.1 For what purpose was the dataset created?

The ADS-Unified-v4 dataset was created to support research in multi-stakeholder curriculum evaluation. It enables benchmarking of multi-objective optimization methods that balance competing interests in educational planning: market demands (employer needs), institutional missions, learner outcomes, and university program requirements.

### 1.2 Who created the dataset and on behalf of which entity?

The dataset was created by the ADS Research Team for academic research purposes, targeting the KDD 2026 Datasets & Benchmarks Track.

### 1.3 Who funded the creation of the dataset?

This research was conducted as part of academic work without dedicated external funding for data collection.

---

## 2. Composition

### 2.1 What do the instances represent?

The dataset contains **28,046 artifacts** of the following types:

| Type | Count | Description |
|------|-------|-------------|
| COURSE | 27,636 | University course descriptions with learning outcomes |
| JOB_ROLE | 1,016 | O*NET occupation descriptions |
| MISSION | 56 | University/college mission statements |
| OUTCOME_MAJOR | 217 | College Scorecard program-level earnings data |

### 2.2 How many instances are there?

- **27,636 courses** from 7 universities across 3 countries
- **1,016 occupations** from O*NET database
- **56 mission statements** from institutional sources
- **217 programs** with earnings data from College Scorecard

### 2.3 University breakdown

| University | Country | Courses |
|------------|---------|---------|
| UC Berkeley | USA | 11,113 |
| MIT | USA | 6,944 |
| KTH | Sweden | 4,768 |
| Stanford | USA | 2,663 |
| UIUC | USA | 860 |
| Cornell | USA | 842 |
| Edinburgh | UK | 446 |

### 2.4 Does the dataset contain sensitive information?

No. The dataset contains only:
- Publicly available course catalog information
- Public government data (O*NET, College Scorecard)
- Published institutional mission statements

No personally identifiable information (PII) is included.

### 2.5 Is it possible to identify individuals?

No. All data is aggregated at the course, program, or institution level.

---

## 3. Collection Process

### 3.1 How was the data acquired?

| Source | Method | License |
|--------|--------|---------|
| MIT OpenCourseWare | Public API/scraping | CC BY-NC-SA 4.0 |
| UC Berkeley Catalog | Public course catalog | Public information |
| Stanford Catalog | Public course catalog | Public information |
| UIUC Catalog | Public course catalog | Public information |
| Cornell Catalog | Public course catalog | Public information |
| KTH KOPPS API | Public API (api.kth.se) | Public information |
| Edinburgh DRPS | Public web scraping | Public information |
| O*NET | Public download | Public Domain (US Gov) |
| College Scorecard | Public API | Public Domain (US Gov) |

### 3.2 What mechanisms were used for data collection?

- **Python scripts** for API access and web scraping
- **Manual curation** for mission statements
- **Automated validation** for data quality

### 3.3 Who was involved in data collection?

The ADS Research Team conducted all data collection. No crowdsourcing was used.

### 3.4 Over what timeframe was data collected?

Data was collected between January and February 2026. Course catalogs reflect the 2025-2026 academic year.

### 3.5 Were any ethical review processes conducted?

The dataset uses only publicly available information and does not require IRB approval.

---

## 4. Preprocessing/Cleaning/Labeling

### 4.1 What preprocessing was done?

1. **Text extraction**: HTML/PDF content converted to plain text
2. **Deduplication**: Removed duplicate courses across sources
3. **Normalization**: Standardized university names, course codes
4. **Embedding**: SBERT embeddings computed for all text artifacts

### 4.2 Was the raw data saved?

Yes. Raw HTML/text snapshots are stored locally with SHA256 hashes for reproducibility. The processed dataset includes `source_hash` fields linking back to raw data.

### 4.3 Is the software for preprocessing available?

Yes. All ingestion scripts are included in the `packages/ads_core/ingest/` directory.

---

## 5. Uses

### 5.1 What are the intended uses?

- **Multi-objective curriculum recommendation**: Finding Pareto-optimal courses
- **Educational data mining**: Analyzing course-market alignment
- **Benchmark evaluation**: Comparing MO optimization methods

### 5.2 What are NOT appropriate uses?

- Automated admission decisions
- Student tracking or surveillance
- Commercial course recommendation without attribution

### 5.3 Has the dataset been used already?

Yes, for the KDD 2026 paper "Agent-Didactic Spaces: Multi-Stakeholder Curriculum Evaluation".

---

## 6. Distribution

### 6.1 How is the dataset distributed?

- **Primary**: Zenodo (archival, DOI-enabled)
- **Mirror**: GitHub repository
- **Format**: CSV + JSONL with Croissant metadata

### 6.2 When will the dataset be released?

Upon paper acceptance (expected mid-2026).

### 6.3 What license is the dataset under?

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

This dataset incorporates:
- O*NET data (Public Domain, U.S. Department of Labor)
- MIT OCW content (CC BY-NC-SA 4.0, with attribution)
- University course catalogs (Public information)

---

## 7. Maintenance

### 7.1 Who supports/maintains the dataset?

The ADS Research Team. Contact: [GitHub Issues](https://github.com/ads-research/ads)

### 7.2 How can users contribute?

- Report issues via GitHub
- Submit pull requests for new university data
- Propose benchmark extensions

### 7.3 Will the dataset be updated?

Updates planned annually to reflect new academic years and additional universities.

### 7.4 Are older versions available?

Yes. Previous versions (unified_v1, v2, v3) are archived in the repository.

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ads_unified_v4_2026,
  title = {ADS-Unified-v4: Multi-Stakeholder Curriculum Evaluation Benchmark},
  author = {ADS Research Team},
  year = {2026},
  publisher = {Zenodo},
  doi = {[TO BE ASSIGNED]},
  url = {https://github.com/ads-research/ads}
}
```

---

## Changelog

- **v4.0.0** (2026-02-06): Added KTH (Sweden) and Edinburgh (UK), 7 universities total
- **v3.0.0** (2026-01-20): Added Stanford, UIUC, Cornell (5 US universities)
- **v2.0.0** (2026-01-15): Added College Scorecard earnings data
- **v1.0.0** (2026-01-10): Initial release (MIT + UC Berkeley)
