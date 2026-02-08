# Datasheet: ADS-Unified-v4

Following the framework of [Gebru et al. (2021)](https://arxiv.org/abs/1803.09010), "Datasheets for Datasets."

## Motivation

### For what purpose was the dataset created?

ADS-Unified-v4 was created to benchmark multi-stakeholder curriculum evaluation systems. Existing education datasets focus on single objectives (e.g., student grades, employment rates). This dataset enables research on *Pareto-optimal* curricula that simultaneously satisfy market demands (O\*NET occupations), institutional missions, university identity, and learner needs. It was created for the KDD 2026 Datasets & Benchmarks Track submission.

### Who created the dataset and on behalf of which entity?

The ADS Research Team created this dataset as part of the Agent-Didactic Spaces research project.

### Who funded the creation of the dataset?

This is an academic research project. No external funding was received specifically for dataset creation.

## Composition

### What do the instances that comprise the dataset represent?

The dataset contains four types of artifacts:

1. **Courses** (27,586): University course descriptions with metadata (title, description, department, prerequisites, level, credit units) from 7 universities across 3 countries.
2. **Job Roles** (1,016): O\*NET occupation profiles from the U.S. Department of Labor, including skill requirements, task descriptions, and technology/tools.
3. **Missions** (62): Institutional mission statements from the included universities and related organizations.
4. **Programs** (398): College Scorecard program-level outcome data.

Additionally:
- **O\*NET structured data**: 71,520 skill records, 13,326 task records, and 32,773 technology records linked to occupations.
- **Crosswalks**: CIP-2020 to SOC-2018 mappings (6,097 records in each direction).
- **Time-travel data**: 4 O\*NET database versions (2022-2025) for temporal drift analysis.

### How many instances are there in total?

28,633 artifacts in the main file (`artifacts.jsonl`), with 27,586 deduplicated course records in `courses.csv`.

### Does the dataset contain all possible instances or is it a sample?

It is a **curated sample**. The 7 universities were selected for diversity:
- **Scale**: UC Berkeley (11K courses) vs. Edinburgh (446 courses)
- **Geography**: 5 US institutions, 1 Swedish (KTH), 1 British (Edinburgh)
- **Type**: Public research universities (all), with both STEM-heavy (MIT, KTH) and comprehensive (UC Berkeley, Edinburgh) institutions

O\*NET data covers the complete U.S. occupational taxonomy (1,016 occupations in SOC-2018).

### What data does each instance consist of?

**Course records** (`courses.csv`):
- `course_id`: Unique identifier (e.g., `6.001`)
- `course_name`: Human-readable title
- `description`: Full text description, typically 50-500 words
- `units`: Credit units (format varies by university)
- `department`: Department or school name
- `level`: Undergraduate, graduate, or unspecified
- `prerequisites`: Free-text prerequisite description
- `university`: Source university name

**Artifact records** (`artifacts.jsonl`):
- `artifact_id`: Namespaced ID (e.g., `course:MIT:8ba94312eb3d`)
- `type`: One of `COURSE`, `JOB_ROLE`, `MISSION`
- `text`: Full text content
- `source_url`: Original URL (where available)
- `source_hash`: SHA-256 of source text for integrity verification
- `timestamp`: ISO 8601 ingestion timestamp

### Is there a label or target associated with each instance?

No. This is an unsupervised benchmark. The multi-objective scores (market, mission, university, learner) are *computed* from embedding similarities, not annotated labels.

### Is any information missing from individual instances?

- `source_url` is empty for some courses where the catalog URL was not available at scrape time.
- `prerequisites` and `units` may be empty for international universities (KTH, Edinburgh) where this metadata was not structured in the source catalog.
- Course `level` classification is imputed for some universities based on course numbering conventions.

### Are there any errors, sources of noise, or redundancies?

- **Cross-listed courses**: Some UC Berkeley courses appear under multiple departments (e.g., `BUDDSTD C119` and `CYPLAN C117`). The `courses.csv` is deduplicated by `course_id`, retaining the first occurrence. The `artifacts.jsonl` has additional text-level deduplication (27,555 unique course artifacts vs. 27,586 unique course_ids).
- **Description length variance**: MIT OCW descriptions average ~200 words; KTH descriptions average ~80 words due to catalog structure differences.
- **Translation artifacts**: Edinburgh and KTH descriptions are in English, but some KTH course names retain Swedish formatting conventions.

### Does the dataset contain data that might be considered confidential?

No. All data is from publicly available sources: published course catalogs, the public O\*NET database, and the College Scorecard (a U.S. government open dataset).

## Collection Process

### How was the data associated with each instance acquired?

- **MIT**: Scraped from MIT OpenCourseWare (ocw.mit.edu) and the MIT course catalog. License: CC BY-NC-SA 4.0.
- **UC Berkeley**: Scraped from the Berkeley Academic Guide (guide.berkeley.edu). Public information.
- **Stanford, UIUC, Cornell**: Scraped from public course catalogs. Public information.
- **KTH**: Scraped from the KTH course catalogue (www.kth.se). Public information.
- **Edinburgh**: Scraped from the University of Edinburgh course finder. Public information.
- **O\*NET**: Downloaded from the O\*NET Resource Center (onetonline.org). Public domain (U.S. Government work).
- **College Scorecard**: Downloaded from collegescorecard.ed.gov. Public domain (U.S. Government work).

### What mechanisms or procedures were used to collect the data?

Custom Python ingestion pipelines (`packages/ads_core/ingest/`) with:
- Per-source parsers for HTML catalog pages
- Content hashing for deduplication
- Provenance tracking (source URL, timestamp, raw text hash)

### Who was involved in the data collection process?

The ADS Research Team. No crowdsourcing or human annotation was used for the core dataset.

### Over what timeframe was the data collected?

- **Course catalogs**: Scraped between January and February 2026 (reflecting 2025-2026 academic year offerings)
- **O\*NET**: Version 30.0 (released 2025) as primary; versions 26.3-28.1 (2022-2023) for time-travel analysis
- **College Scorecard**: 2024 release

### Were any ethical review processes conducted?

No IRB review was required as all data is publicly available institutional information with no personally identifiable information.

## Preprocessing/Cleaning/Labeling

### Was any preprocessing/cleaning/labeling of the data done?

1. **Text normalization**: HTML stripped, Unicode normalized (NFKD), excessive whitespace collapsed.
2. **Deduplication**: Course records deduplicated by `course_id` (removed 50 cross-listed duplicates). Artifact records additionally deduplicated by text content hash (27,555 unique text artifacts from 27,586 unique IDs).
3. **Field extraction**: Structured fields (department, level, units, prerequisites) extracted from free-text catalog entries using regex patterns specific to each university's catalog format.
4. **O\*NET processing**: Raw tab-delimited O\*NET files processed into per-occupation CSVs with skill/task/technology records.
5. **Crosswalk construction**: CIP-SOC crosswalk extracted from the official NCES Excel file.

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

Yes. Raw HTML pages, O\*NET database files, and original catalog downloads are preserved in `data/raw/` in the source repository.

### Is the software that was used to preprocess/clean/label the data available?

Yes. All ingestion code is in `packages/ads_core/ingest/` in the ADS repository. The deduplication script is reproducible from the provided data.

## Uses

### What are the primary intended uses?

1. **Benchmarking** multi-objective curriculum optimization algorithms
2. **Evaluating** embedding-based alignment between educational content and labor market demands
3. **Studying** cross-national curriculum design patterns
4. **Analyzing** temporal drift in labor market skill requirements (time-travel data)

### What are some non-intended but foreseeable uses?

- Course recommendation systems
- Labor market analysis independent of education
- NLP research on educational text (description embeddings, prerequisite parsing)

### Are there tasks for which the dataset should not be used?

- **Individual student assessment**: This dataset contains no student data and should not be used to evaluate individual learning outcomes.
- **Automated curriculum replacement**: The Pareto analysis is a decision-support tool, not a prescription. Curriculum decisions require human judgment.
- **Cross-country policy comparison**: The university selection is not representative of national education systems.

## Distribution

### How will the dataset be distributed?

- **HuggingFace Hub**: `ads-research/ads-benchmark-v4` (primary distribution)
- **Zenodo**: Archived with DOI for permanent citation
- **GitHub**: Code and scripts (data via HF/Zenodo links)

### When was the dataset first released?

February 2026 (KDD 2026 submission).

### What license is the dataset distributed under?

Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE` for details and per-source attribution.

### Are there any restrictions on use?

MIT OCW content is originally CC BY-NC-SA 4.0. Our use is transformative (metadata extraction, not content redistribution). Users who extract full course descriptions for commercial purposes should verify compliance with MIT OCW's license.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

The ADS Research Team maintains the dataset. Updates will be published on HuggingFace Hub and Zenodo.

### How can the owner/curator/manager of the dataset be contacted?

Via the GitHub repository issues page.

### Will the dataset be updated?

Potentially. Future versions may add:
- Additional universities (particularly from Asia, Africa, and South America)
- Updated O\*NET versions as released
- Richer course metadata (learning outcomes, assessment methods)

### If the dataset relates to people, are there applicable limits on data retention?

The dataset does not contain data about identifiable people. Course descriptions are institutional products. Mission statements are public organizational documents.

### Will older versions of the dataset continue to be available?

Yes. Each version is archived on Zenodo with a unique DOI. The unified_v3 dataset (7 US universities, no international) remains available in the repository.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daume III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86-92.
- O\*NET Resource Center: https://www.onetonline.org/
- College Scorecard: https://collegescorecard.ed.gov/
