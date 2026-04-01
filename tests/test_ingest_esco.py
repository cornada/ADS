"""Tests for ESCO ingestion. Mock-based — no real API calls."""
from __future__ import annotations

import json
from unittest.mock import patch

SAMPLE_OCCUPATIONS = [
    {
        "uri": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
        "title": "software developer",
        "isco_group": "Software developers",
        "isco_code": "2512",
        "description": "Software developers design and implement software systems.",
        "essential_skills": [
            {"uri": "http://data.europa.eu/esco/skill/aaa", "title": "programming", "skill_type": ""},
            {"uri": "http://data.europa.eu/esco/skill/bbb", "title": "software testing", "skill_type": ""},
        ],
        "optional_skills": [
            {"uri": "http://data.europa.eu/esco/skill/ccc", "title": "cloud computing", "skill_type": ""},
        ],
        "essential_skill_count": 2,
        "optional_skill_count": 1,
    },
    {
        "uri": "http://data.europa.eu/esco/occupation/aabbccdd-1234-5678-abcd-111122223333",
        "title": "data scientist",
        "isco_group": "Database and network professionals",
        "isco_code": "2521",
        "description": "Data scientists analyze complex data.",
        "essential_skills": [
            {"uri": "http://data.europa.eu/esco/skill/ddd", "title": "programming", "skill_type": ""},
            {"uri": "http://data.europa.eu/esco/skill/eee", "title": "machine learning", "skill_type": ""},
        ],
        "optional_skills": [],
        "essential_skill_count": 2,
        "optional_skill_count": 0,
    },
]


class TestESCOSaveOccupations:

    def test_save_occupations_creates_file(self, tmp_path):
        from ads_core.ingest.esco import save_occupations

        out = save_occupations(SAMPLE_OCCUPATIONS, tmp_path / "occ.csv")
        assert out.exists()

    def test_save_occupations_row_count(self, tmp_path):
        import csv
        from ads_core.ingest.esco import save_occupations

        out = save_occupations(SAMPLE_OCCUPATIONS, tmp_path / "occ.csv")
        with out.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["title"] == "software developer"


class TestESCOSkillAggregation:

    def test_aggregate_counts(self):
        from ads_core.ingest.esco import aggregate_skills

        skills = aggregate_skills(SAMPLE_OCCUPATIONS)
        skill_map = {s["title"]: s for s in skills}

        # "programming" appears in both occupations
        assert skill_map["programming"]["occupation_count"] == 2
        assert skill_map["machine learning"]["occupation_count"] == 1
        assert skill_map["cloud computing"]["occupation_count"] == 1

    def test_aggregate_reuse_level(self):
        from ads_core.ingest.esco import aggregate_skills

        skills = aggregate_skills(SAMPLE_OCCUPATIONS)
        skill_map = {s["title"]: s for s in skills}

        assert skill_map["programming"]["reuse_level"] == "essential"
        assert skill_map["cloud computing"]["reuse_level"] == "optional"


class TestESCOArtifacts:

    def test_generate_artifacts_count(self, tmp_path):
        from ads_core.ingest.esco import generate_artifacts

        out = generate_artifacts(SAMPLE_OCCUPATIONS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            artifacts = [json.loads(line) for line in f]
        assert len(artifacts) == 2

    def test_artifact_structure(self, tmp_path):
        from ads_core.ingest.esco import generate_artifacts

        out = generate_artifacts(SAMPLE_OCCUPATIONS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert art["artifact_id"].startswith("esco:")
        assert art["type"] == "JOB_ROLE"
        assert "software developer" in art["text"]
        assert art["metadata"]["institution"] == "MARKET_EU"
        assert art["metadata"]["country"] == "EU"
        assert art["metadata"]["isco_code"] == "2512"

    def test_artifact_contains_skills(self, tmp_path):
        from ads_core.ingest.esco import generate_artifacts

        out = generate_artifacts(SAMPLE_OCCUPATIONS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert "programming" in art["text"]
        assert "software testing" in art["text"]

    def test_artifact_provenance(self, tmp_path):
        from ads_core.ingest.esco import generate_artifacts

        out = generate_artifacts(SAMPLE_OCCUPATIONS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        prov = art["metadata"]["provenance"]
        assert prov["dataset"] == "esco"
        assert "fetched_at" in prov


class TestESCOAPI:

    @patch("ads_core.ingest.esco._api_get")
    def test_search_occupations(self, mock_api):
        from ads_core.ingest.esco import search_occupations

        mock_api.return_value = {
            "total": 1,
            "_embedded": {
                "results": [
                    {
                        "uri": "http://data.europa.eu/esco/occupation/test123",
                        "title": "Test Occupation",
                    }
                ]
            },
        }

        results = search_occupations("test")
        assert len(results) == 1
        assert results[0]["title"] == "Test Occupation"

    @patch("ads_core.ingest.esco._api_get")
    def test_fetch_occupation_with_skills(self, mock_api):
        from ads_core.ingest.esco import fetch_occupation

        mock_api.return_value = {
            "title": "Test Dev",
            "description": {"en": "A test developer role."},
            "_links": {
                "broaderIscoGroup": [{"code": "2512", "title": "Software devs"}],
                "hasEssentialSkill": [
                    {"uri": "http://skill/1", "title": "Python", "skillType": "skill"},
                ],
                "hasOptionalSkill": [
                    {"uri": "http://skill/2", "title": "Docker", "skillType": "skill"},
                ],
            },
        }

        occ = fetch_occupation("http://test/occ")
        assert occ["title"] == "Test Dev"
        assert occ["isco_code"] == "2512"
        assert len(occ["essential_skills"]) == 1
        assert occ["essential_skills"][0]["title"] == "Python"
        assert len(occ["optional_skills"]) == 1
        assert occ["description"] == "A test developer role."
