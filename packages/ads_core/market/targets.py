"""Market target definitions based on O*NET/ESCO taxonomy.

Defines canonical market categories for comparing predicted market-fit
with observed career outcomes. Categories are aligned with:
- O*NET occupation groups (https://www.onetonline.org/)
- ESCO skills/occupations (https://esco.ec.europa.eu/)
- BLS occupational classifications

Category Design Principles:
1. Broad enough to capture major career paths
2. Distinct enough for meaningful differentiation
3. Aligned with typical FDS/outcomes survey categories
4. Based on open/public taxonomies (no proprietary data)

Version: 1.0
Last updated: 2026-01-15
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class MarketCategory:
    """A canonical market category for outcome validation."""

    category_id: str
    name: str
    description: str
    onet_groups: List[str]  # O*NET occupation group codes
    keywords: List[str]     # Keywords for keyword baseline
    typical_majors: List[str]  # Expected majors (for reference, not used in model)

    def to_text(self) -> str:
        """Generate text representation for embedding."""
        parts = [
            f"{self.name}: {self.description}",
            f"Career paths in {self.name.lower()} include: {', '.join(self.keywords[:5])}.",
        ]
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category_id": self.category_id,
            "name": self.name,
            "description": self.description,
            "onet_groups": self.onet_groups,
            "keywords": self.keywords,
            "typical_majors": self.typical_majors,
        }


# Canonical market categories (10 categories for MVP)
# Based on O*NET major groups and typical FDS outcome categories
MARKET_CATEGORIES: Dict[str, MarketCategory] = {
    "software_tech": MarketCategory(
        category_id="software_tech",
        name="Software & Technology",
        description="Software development, engineering, and technology roles including "
                    "web development, mobile apps, systems programming, and DevOps.",
        onet_groups=["15-1251", "15-1252", "15-1253", "15-1254", "15-1255"],
        keywords=[
            "software engineer", "software developer", "web developer",
            "full stack", "backend", "frontend", "mobile developer",
            "devops", "site reliability", "systems programmer",
            "application developer", "platform engineer", "coding",
        ],
        typical_majors=["Computer Science", "Software Engineering", "Computer Engineering"],
    ),

    "data_ml": MarketCategory(
        category_id="data_ml",
        name="Data Science & Machine Learning",
        description="Data analysis, machine learning, artificial intelligence, and "
                    "analytics roles including data scientists, ML engineers, and analysts.",
        onet_groups=["15-2051", "15-2099", "15-1221"],
        keywords=[
            "data scientist", "machine learning", "data analyst",
            "ml engineer", "ai engineer", "deep learning",
            "data engineer", "analytics", "business intelligence",
            "statistical modeling", "predictive analytics", "nlp",
        ],
        typical_majors=["Data Science", "Statistics", "Computer Science", "Mathematics"],
    ),

    "finance_business": MarketCategory(
        category_id="finance_business",
        name="Finance & Business",
        description="Finance, accounting, consulting, and business operations roles "
                    "including investment banking, financial analysis, and management consulting.",
        onet_groups=["13-2051", "13-2052", "13-2053", "13-1111", "11-3031"],
        keywords=[
            "financial analyst", "investment banking", "consulting",
            "management consultant", "accountant", "auditor",
            "business analyst", "strategy", "private equity",
            "venture capital", "corporate finance", "operations",
        ],
        typical_majors=["Business Administration", "Finance", "Economics", "Accounting"],
    ),

    "research_academia": MarketCategory(
        category_id="research_academia",
        name="Research & Academia",
        description="Academic research, graduate studies, and research scientist roles "
                    "in universities, research institutions, and R&D departments.",
        onet_groups=["19-1099", "19-2099", "19-3099", "25-1099"],
        keywords=[
            "research scientist", "postdoc", "professor",
            "graduate student", "phd", "research assistant",
            "lab researcher", "principal investigator", "academic",
            "research fellow", "scientist", "scholar",
        ],
        typical_majors=["Physics", "Chemistry", "Biology", "Mathematics", "Psychology"],
    ),

    "healthcare_medical": MarketCategory(
        category_id="healthcare_medical",
        name="Healthcare & Medical",
        description="Healthcare, medical, nursing, and public health roles including "
                    "clinical positions, healthcare administration, and medical research.",
        onet_groups=["29-1141", "29-1171", "29-2061", "31-9091", "21-1094"],
        keywords=[
            "nurse", "physician", "healthcare", "medical",
            "clinical", "hospital", "patient care", "nursing",
            "public health", "health administration", "therapist",
            "pharmacist", "medical researcher", "healthcare analyst",
        ],
        typical_majors=["Nursing", "Public Health", "Biology", "Health Sciences"],
    ),

    "engineering_manufacturing": MarketCategory(
        category_id="engineering_manufacturing",
        name="Engineering & Manufacturing",
        description="Engineering roles in manufacturing, aerospace, automotive, and "
                    "industrial sectors including mechanical, electrical, and civil engineering.",
        onet_groups=["17-2011", "17-2041", "17-2071", "17-2081", "17-2141"],
        keywords=[
            "mechanical engineer", "electrical engineer", "civil engineer",
            "aerospace engineer", "manufacturing engineer", "design engineer",
            "project engineer", "structural engineer", "systems engineer",
            "automotive engineer", "industrial engineer", "process engineer",
        ],
        typical_majors=["Mechanical Engineering", "Electrical Engineering", "Civil Engineering"],
    ),

    "creative_media": MarketCategory(
        category_id="creative_media",
        name="Creative & Media",
        description="Creative, design, media, and communications roles including "
                    "journalism, marketing, graphic design, and content creation.",
        onet_groups=["27-1024", "27-3031", "27-3043", "27-4021", "27-1014"],
        keywords=[
            "graphic designer", "journalist", "content creator",
            "marketing", "communications", "public relations",
            "copywriter", "media producer", "art director",
            "ux designer", "brand manager", "social media",
        ],
        typical_majors=["Journalism", "Communications", "Graphic Design", "Marketing"],
    ),

    "education_teaching": MarketCategory(
        category_id="education_teaching",
        name="Education & Teaching",
        description="Education, teaching, and training roles including K-12 teachers, "
                    "instructional designers, and educational administrators.",
        onet_groups=["25-2021", "25-2022", "25-2031", "25-3099", "25-9031"],
        keywords=[
            "teacher", "educator", "instructor", "professor",
            "curriculum developer", "instructional designer",
            "education administrator", "academic advisor",
            "tutor", "teaching assistant", "school counselor",
        ],
        typical_majors=["Education", "English", "Mathematics", "Science Education"],
    ),

    "law_policy": MarketCategory(
        category_id="law_policy",
        name="Law & Policy",
        description="Legal, policy, government, and regulatory roles including "
                    "lawyers, policy analysts, and government positions.",
        onet_groups=["23-1011", "23-1021", "23-2011", "13-1041", "19-3094"],
        keywords=[
            "lawyer", "attorney", "paralegal", "legal counsel",
            "policy analyst", "government affairs", "lobbyist",
            "regulatory affairs", "public policy", "legislative",
            "compliance", "legal researcher", "public administration",
        ],
        typical_majors=["Political Science", "Public Policy", "Law", "Government"],
    ),

    "environmental_sustainability": MarketCategory(
        category_id="environmental_sustainability",
        name="Environmental & Sustainability",
        description="Environmental science, sustainability, and conservation roles "
                    "including environmental consultants, sustainability managers, and conservation scientists.",
        onet_groups=["19-2041", "19-1031", "17-2081", "19-4042", "13-1199"],
        keywords=[
            "environmental scientist", "sustainability manager",
            "conservation", "environmental consultant", "ecologist",
            "climate analyst", "renewable energy", "green building",
            "environmental engineer", "natural resources", "wildlife",
        ],
        typical_majors=["Environmental Science", "Environmental Engineering", "Biology", "Sustainability"],
    ),
}


def get_market_category_texts() -> Dict[str, str]:
    """Get text representations of all market categories.

    Returns:
        Dict mapping category_id to text for embedding
    """
    return {cat_id: cat.to_text() for cat_id, cat in MARKET_CATEGORIES.items()}


def create_market_target_vectors(
    encoder_fn: Callable[[List[str]], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Create embedding vectors for all market categories.

    Args:
        encoder_fn: Function that encodes texts to vectors

    Returns:
        Dict mapping category_id to embedding vector
    """
    texts = get_market_category_texts()
    category_ids = list(texts.keys())
    embeddings = encoder_fn(list(texts.values()))

    return {cat_id: embeddings[i] for i, cat_id in enumerate(category_ids)}


def compute_market_similarity(
    unit_vector: np.ndarray,
    market_vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute similarity of a unit vector to all market categories.

    Args:
        unit_vector: Embedding vector for a unit (major/program)
        market_vectors: Dict of category_id to embedding vector

    Returns:
        Dict of category_id to similarity score [0, 1]
    """
    similarities = {}

    for cat_id, cat_vec in market_vectors.items():
        # Cosine similarity
        dot = np.dot(unit_vector, cat_vec)
        norm_u = np.linalg.norm(unit_vector)
        norm_c = np.linalg.norm(cat_vec)

        if norm_u > 1e-8 and norm_c > 1e-8:
            sim = dot / (norm_u * norm_c)
        else:
            sim = 0.0

        # Clamp to [0, 1] (similarity can be negative for unrelated texts)
        similarities[cat_id] = max(0.0, min(1.0, (sim + 1) / 2))

    return similarities


def predict_market_distribution(
    unit_vector: np.ndarray,
    market_vectors: Dict[str, np.ndarray],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Predict probability distribution over market categories.

    Uses softmax over similarities to produce a normalized distribution.

    Args:
        unit_vector: Embedding vector for a unit
        market_vectors: Dict of category_id to embedding vector
        temperature: Softmax temperature (higher = more uniform)

    Returns:
        Dict of category_id to probability [0, 1], summing to 1
    """
    similarities = compute_market_similarity(unit_vector, market_vectors)

    # Convert to numpy array for softmax
    cat_ids = list(similarities.keys())
    scores = np.array([similarities[cat_id] for cat_id in cat_ids])

    # Softmax with temperature
    exp_scores = np.exp((scores - scores.max()) / temperature)
    probs = exp_scores / exp_scores.sum()

    return {cat_id: float(probs[i]) for i, cat_id in enumerate(cat_ids)}


# Mapping from major keywords to expected market categories
# Used for the keyword baseline (B0)
MAJOR_TO_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "software_tech": [
        "computer science", "software", "programming", "cs",
        "computing", "informatics", "computer engineering",
    ],
    "data_ml": [
        "data science", "statistics", "machine learning", "analytics",
        "data analytics", "applied math", "quantitative",
    ],
    "finance_business": [
        "business", "finance", "economics", "accounting",
        "management", "mba", "commerce", "entrepreneurship",
    ],
    "research_academia": [
        "physics", "chemistry", "biology", "biochemistry",
        "neuroscience", "cognitive science", "research",
    ],
    "healthcare_medical": [
        "nursing", "health", "medical", "public health",
        "nutrition", "kinesiology", "pre-med", "pharmacy",
    ],
    "engineering_manufacturing": [
        "mechanical engineering", "electrical engineering",
        "civil engineering", "aerospace", "materials",
        "industrial engineering", "bioengineering",
    ],
    "creative_media": [
        "journalism", "media", "communications", "film",
        "art", "design", "graphic", "advertising",
    ],
    "education_teaching": [
        "education", "teaching", "pedagogy", "curriculum",
        "childhood", "elementary", "secondary",
    ],
    "law_policy": [
        "political science", "policy", "government", "law",
        "international relations", "public administration",
    ],
    "environmental_sustainability": [
        "environmental", "ecology", "sustainability",
        "conservation", "natural resources", "climate",
    ],
}


def keyword_baseline_prediction(
    major_name: str,
) -> Dict[str, float]:
    """Predict market category using keyword matching (baseline B0).

    Args:
        major_name: Name of the major/program

    Returns:
        Dict of category_id to score [0, 1]
    """
    major_lower = major_name.lower()
    scores = {}

    for cat_id, keywords in MAJOR_TO_CATEGORY_KEYWORDS.items():
        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in major_lower)
        # Normalize by number of keywords
        scores[cat_id] = min(1.0, matches / 2)  # Cap at 2 matches = 1.0

    # Normalize to sum to 1 if any matches
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        # Uniform distribution if no matches
        n = len(MAJOR_TO_CATEGORY_KEYWORDS)
        scores = {k: 1.0 / n for k in MAJOR_TO_CATEGORY_KEYWORDS}

    return scores
