"""Tests for Russian labor market ingestors: profstandart and headhunter.

Mock-based tests — no real API calls or file downloads.
"""
from __future__ import annotations

import csv
import json
from unittest.mock import patch


# ============================================================================
# Profstandart
# ============================================================================

SAMPLE_STANDARDS = [
    {
        "ps_code": "06.001",
        "name": "Программист",
        "area": "Связь, информационные и коммуникационные технологии",
        "activity_type": "Разработка программного обеспечения",
        "reg_number": "4",
        "order_date": "2013-11-18",
        "responsible_org": "СПК ИТ",
    },
    {
        "ps_code": "22.001",
        "name": "Металлург",
        "area": "Металлургическое производство",
        "activity_type": "Производство черных и цветных металлов",
        "reg_number": "15",
        "order_date": "2014-03-10",
        "responsible_org": "СПК горно-металлургической отрасли",
    },
]


class TestProfstandartCSV:

    def test_save_csv_creates_file(self, tmp_path):
        from ads_core.ingest.profstandart import save_csv, FIELDNAMES

        out = save_csv(SAMPLE_STANDARDS, tmp_path / "test.csv")
        assert out.exists()

        with out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert set(reader.fieldnames) == set(FIELDNAMES)
        assert rows[0]["ps_code"] == "06.001"
        assert rows[1]["name"] == "Металлург"

    def test_save_csv_columns_match(self, tmp_path):
        from ads_core.ingest.profstandart import save_csv, FIELDNAMES

        out = save_csv(SAMPLE_STANDARDS, tmp_path / "test.csv")
        with out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == FIELDNAMES


class TestProfstandartArtifacts:

    def test_generate_artifacts_count(self, tmp_path):
        from ads_core.ingest.profstandart import generate_artifacts

        out = generate_artifacts(SAMPLE_STANDARDS, tmp_path / "artifacts.jsonl")
        assert out.exists()

        with out.open("r", encoding="utf-8") as f:
            artifacts = [json.loads(line) for line in f]

        assert len(artifacts) == 2

    def test_artifact_structure(self, tmp_path):
        from ads_core.ingest.profstandart import generate_artifacts

        out = generate_artifacts(SAMPLE_STANDARDS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert art["artifact_id"] == "profstandart:06.001"
        assert art["type"] == "JOB_ROLE"
        assert "Программист" in art["text"]
        assert art["metadata"]["institution"] == "MARKET_RU"
        assert art["metadata"]["country"] == "RU"
        assert art["metadata"]["ps_code"] == "06.001"

    def test_artifact_provenance(self, tmp_path):
        from ads_core.ingest.profstandart import generate_artifacts

        out = generate_artifacts(SAMPLE_STANDARDS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        prov = art["metadata"]["provenance"]
        assert prov["dataset"] == "profstandart"
        assert "raw_text_hash" in prov
        assert "fetched_at" in prov

    def test_artifact_text_contains_area(self, tmp_path):
        from ads_core.ingest.profstandart import generate_artifacts

        out = generate_artifacts(SAMPLE_STANDARDS, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert "Связь, информационные" in art["text"]
        assert "Professional standard code: 06.001" in art["text"]


class TestProfstandartAreaMapping:

    def test_area_to_fgos_prefix(self):
        from ads_core.ingest.profstandart import _AREA_TO_FGOS_PREFIX

        # IT maps to FGOS 09, 10, 11
        it_area = "Связь, информационные и коммуникационные технологии"
        assert "09" in _AREA_TO_FGOS_PREFIX[it_area]

        # Metallurgy maps to FGOS 22
        met_area = "Металлургическое производство"
        assert "22" in _AREA_TO_FGOS_PREFIX[met_area]


# ============================================================================
# HeadHunter
# ============================================================================

SAMPLE_VACANCIES = [
    {
        "vacancy_id": "12345",
        "name": "Python-разработчик",
        "salary_from": 150000,
        "salary_to": 250000,
        "salary_currency": "RUR",
        "experience": "between1And3",
        "schedule": "remote",
        "employment": "full",
        "area": "Москва",
        "employer_name": "Яндекс",
        "professional_roles": "Программист, разработчик",
        "key_skills": "Python,Django,PostgreSQL",
        "published_at": "2026-02-01",
        "url": "https://hh.ru/vacancy/12345",
        "search_query": "DevOps",
    },
    {
        "vacancy_id": "67890",
        "name": "Data Scientist",
        "salary_from": 200000,
        "salary_to": 350000,
        "salary_currency": "RUR",
        "experience": "between3And6",
        "schedule": "fullDay",
        "employment": "full",
        "area": "Санкт-Петербург",
        "employer_name": "Сбер",
        "professional_roles": "Data Scientist",
        "key_skills": "Python,PyTorch,Machine Learning",
        "published_at": "2026-02-05",
        "url": "https://hh.ru/vacancy/67890",
        "search_query": "data scientist",
    },
]


class TestHeadHunterCSV:

    def test_save_vacancies(self, tmp_path):
        from ads_core.ingest.headhunter import save_vacancies

        out = save_vacancies(SAMPLE_VACANCIES, tmp_path / "vacancies.csv")
        assert out.exists()

        with out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["vacancy_id"] == "12345"
        assert rows[1]["employer_name"] == "Сбер"

    def test_save_skills(self, tmp_path):
        from ads_core.ingest.headhunter import aggregate_skills, save_skills

        skills = aggregate_skills(SAMPLE_VACANCIES)
        out = save_skills(skills, tmp_path / "skills.csv")
        assert out.exists()

        with out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Python appears in both vacancies
        python_row = next(r for r in rows if r["skill_name"] == "Python")
        assert int(python_row["vacancy_count"]) == 2


class TestHeadHunterSkillAggregation:

    def test_aggregate_counts(self):
        from ads_core.ingest.headhunter import aggregate_skills

        skills = aggregate_skills(SAMPLE_VACANCIES)
        skill_map = {s["skill_name"]: s for s in skills}

        assert skill_map["Python"]["vacancy_count"] == 2
        assert skill_map["Django"]["vacancy_count"] == 1
        assert skill_map["PyTorch"]["vacancy_count"] == 1

    def test_aggregate_salary(self):
        from ads_core.ingest.headhunter import aggregate_skills

        skills = aggregate_skills(SAMPLE_VACANCIES)
        skill_map = {s["skill_name"]: s for s in skills}

        # Python: avg of (150k, 200k) = 175k from, (250k, 350k) = 300k to
        python = skill_map["Python"]
        assert python["avg_salary_from"] == 175000
        assert python["avg_salary_to"] == 300000

    def test_aggregate_queries_tracked(self):
        from ads_core.ingest.headhunter import aggregate_skills

        skills = aggregate_skills(SAMPLE_VACANCIES)
        skill_map = {s["skill_name"]: s for s in skills}

        python = skill_map["Python"]
        assert "DevOps" in python["search_query"]
        assert "data scientist" in python["search_query"]


class TestHeadHunterArtifacts:

    def test_generate_artifacts_count(self, tmp_path):
        from ads_core.ingest.headhunter import generate_artifacts

        out = generate_artifacts(SAMPLE_VACANCIES, tmp_path / "artifacts.jsonl")
        assert out.exists()

        with out.open("r", encoding="utf-8") as f:
            artifacts = [json.loads(line) for line in f]

        assert len(artifacts) == 2

    def test_artifact_structure(self, tmp_path):
        from ads_core.ingest.headhunter import generate_artifacts

        out = generate_artifacts(SAMPLE_VACANCIES, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert art["artifact_id"] == "hh:12345"
        assert art["type"] == "JOB_ROLE"
        assert "Python-разработчик" in art["text"]
        assert art["metadata"]["institution"] == "MARKET_RU"
        assert art["metadata"]["country"] == "RU"
        assert art["metadata"]["vacancy_id"] == "12345"

    def test_artifact_key_skills_list(self, tmp_path):
        from ads_core.ingest.headhunter import generate_artifacts

        out = generate_artifacts(SAMPLE_VACANCIES, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert art["metadata"]["key_skills"] == ["Python", "Django", "PostgreSQL"]

    def test_artifact_salary_preserved(self, tmp_path):
        from ads_core.ingest.headhunter import generate_artifacts

        out = generate_artifacts(SAMPLE_VACANCIES, tmp_path / "artifacts.jsonl")
        with out.open("r", encoding="utf-8") as f:
            art = json.loads(f.readline())

        assert art["metadata"]["salary_from"] == 150000
        assert art["metadata"]["salary_to"] == 250000
        assert art["metadata"]["salary_currency"] == "RUR"


class TestHeadHunterAPI:

    @patch("ads_core.ingest.headhunter._api_get")
    def test_fetch_vacancies_dedup(self, mock_api):
        from ads_core.ingest.headhunter import fetch_vacancies_by_query

        # First call: search results, Second call: full vacancy detail
        mock_api.side_effect = [
            {
                "items": [
                    {"id": "111", "name": "Test"},
                    {"id": "111", "name": "Test"},  # duplicate
                ],
                "pages": 1,
            },
            # full vacancy for id 111
            {
                "id": "111",
                "name": "Test Vacancy",
                "salary": {"from": 100000, "to": 200000, "currency": "RUR"},
                "experience": {"id": "noExperience"},
                "schedule": {"id": "remote"},
                "employment": {"id": "full"},
                "area": {"name": "Москва"},
                "employer": {"name": "TestCo"},
                "professional_roles": [{"name": "Dev"}],
                "key_skills": [{"name": "Python"}, {"name": "Go"}],
                "published_at": "2026-01-01",
                "alternate_url": "https://hh.ru/vacancy/111",
            },
        ]

        results = fetch_vacancies_by_query("test", max_pages=1, delay=0)
        assert len(results) == 1
        assert results[0]["key_skills"] == "Python,Go"
