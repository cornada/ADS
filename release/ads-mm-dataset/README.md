# ADS-MM: A Multimodal Cross-National Educational Resource Dataset

## Overview

ADS-MM is a multimodal educational resource dataset spanning **30,249 courses** from **8 universities** across **4 countries** (USA, Sweden, UK, Russia), augmented with multimedia resource annotations from MIT OpenCourseWare and instructional activity metadata from MISIS.

**Key finding**: Course text descriptions encode multimedia instructional design. The primary axis of the educational embedding space correlates with multimedia resource richness (r=0.41).

## Dataset Scale

| Component | Count |
|-----------|-------|
| Total records | 36,192 |
| Courses | 29,797 |
| O*NET occupations | 5,752 |
| Competency codes | 581 |
| Mission statements | 62 |
| Course images (MIT OCW) | 1,689 |
| Video course transcripts | 221 |

## Universities

| University | Country | Courses | Modalities |
|-----------|---------|---------|------------|
| UC Berkeley | USA | 11,122 | text |
| MIT OCW | USA | 6,955 | text + multimodal + images |
| KTH | Sweden | 4,768 | text |
| MISIS | Russia | 2,756 | text + activity types + competencies |
| Stanford | USA | 2,665 | text |
| UIUC | USA | 862 | text |
| Cornell | USA | 844 | text |
| Edinburgh | UK | 446 | text |

## Files

```
ads_mm_canonical.jsonl     # Unified dataset (36,192 records, all layers)
join_manifest.json         # Schema, statistics, join metadata
LICENSE                    # CC BY-NC-SA 4.0
images/                    # 1,689 MIT OCW course cover images
multimodal/
  ocw_video_metadata.jsonl # MIT OCW resource type annotations (1,689)
  ocw_transcripts.jsonl    # Lecture titles + syllabi (221)
  misis_courses.jsonl      # MISIS hours + competencies (2,694)
baselines/
  baseline_results.json    # TF-IDF baseline results
  sbert_results.json       # SBERT baseline results
  latent_space.json        # Full-scale latent space analysis (36K SBERT)
scripts/
  fetch_ocw_multimodal.py  # Reproducible data collection
  build_canonical_dataset.py
  crossmodal_baselines.py
  latent_space_analysis.py
```

## Canonical Record Schema

Each record in `ads_mm_canonical.jsonl` contains:

```json
{
  "canonical_id": "MIT:6.006",
  "type": "COURSE",
  "institution": "MIT",
  "title": "Introduction to Algorithms",
  "text": "...",
  "metadata": {},
  "multimodal": {
    "has_video": true,
    "has_notes": true,
    "resource_richness": 3,
    "learning_resource_types": ["Lecture Videos", "Lecture Notes", "Problem Sets"]
  },
  "activity": {},
  "competencies": [],
  "prerequisites": [],
  "transcripts": {"lecture_titles": [...], "lecture_count": 26},
  "image": {"path": "images/6-006-...", "available": true}
}
```

## Evaluation Tasks

1. **Richness Prediction**: Predict resource richness (0-6) from text. Best: SBERT rho=0.633.
2. **Resource Type Classification**: Predict video/notes/assessment from text. Best: SBERT macro F1=0.572.
3. **Cross-Institutional Transfer** (open): Train on MIT, predict other universities.

## Latent Space Key Findings

- **Unified didactic space**: separation ratio 0.70 (institutions overlap, not cluster)
- **PC1 encodes multimedia**: r=0.41 correlation with resource richness
- **36% cross-national neighbors**: MIT courses have neighbors from other universities
- **MISIS tightest cluster**: intra-distance 0.54 (FGOS standardization effect)

## Citation

```bibtex
@inproceedings{ads_mm_2026,
  title     = {{ADS-MM}: A Multimodal Cross-National Educational Resource Dataset for Curriculum Analysis},
  author    = {Anonymous},
  booktitle = {Proceedings of ACM Multimedia},
  year      = {2026}
}
```

## License

CC BY-NC-SA 4.0 (see LICENSE file for component details).
