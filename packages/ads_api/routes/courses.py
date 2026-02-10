"""Course API endpoints.

CRUD operations, search, and embedding-based similarity for courses.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import numpy as np

router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class CourseOut(BaseModel):
    """Course response model."""
    course_id: str
    course_name: str
    university: str
    department: str = ""
    level: str = ""
    units: float = 0.0
    description: str = ""
    prerequisites: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CourseSearchRequest(BaseModel):
    """Search courses by text or embedding similarity."""
    query: str = Field(..., min_length=1, description="Search query text")
    university: Optional[str] = None
    level: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class CourseSimilarityRequest(BaseModel):
    """Find courses similar to a given course."""
    course_id: str
    limit: int = Field(default=10, ge=1, le=50)
    same_university: bool = False


class CourseSearchResult(BaseModel):
    """A search result with relevance score."""
    course: CourseOut
    score: float


# ============================================================================
# In-memory store (replaced by DB in production)
# ============================================================================

_course_store: Dict[str, CourseOut] = {}
_embeddings: Dict[str, np.ndarray] = {}


def load_courses(courses: List[Dict[str, Any]], embeddings: Optional[Dict[str, np.ndarray]] = None):
    """Load courses into the store (called at startup)."""
    global _course_store, _embeddings
    _course_store = {}
    for c in courses:
        course = CourseOut(**c)
        _course_store[course.course_id] = course
    if embeddings:
        _embeddings = embeddings


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/", response_model=List[CourseOut])
def list_courses(
    university: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List courses with optional filtering."""
    courses = list(_course_store.values())
    if university:
        courses = [c for c in courses if c.university.lower() == university.lower()]
    if level:
        courses = [c for c in courses if c.level == level]
    return courses[offset:offset + limit]


@router.get("/{course_id}", response_model=CourseOut)
def get_course(course_id: str):
    """Get a specific course by ID."""
    if course_id not in _course_store:
        raise HTTPException(status_code=404, detail=f"Course not found: {course_id}")
    return _course_store[course_id]


@router.post("/search", response_model=List[CourseSearchResult])
def search_courses(req: CourseSearchRequest):
    """Search courses by text query.

    Uses simple text matching. In production, would use
    embedding similarity for semantic search.
    """
    query_lower = req.query.lower()
    results = []

    for course in _course_store.values():
        if req.university and course.university.lower() != req.university.lower():
            continue
        if req.level and course.level != req.level:
            continue

        # Simple text matching score
        text = f"{course.course_name} {course.description} {course.department}".lower()
        words = query_lower.split()
        score = sum(1.0 for w in words if w in text) / max(len(words), 1)

        if score > 0:
            results.append(CourseSearchResult(course=course, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:req.limit]


@router.post("/similar", response_model=List[CourseSearchResult])
def find_similar(req: CourseSimilarityRequest):
    """Find courses similar to a given course using embedding distance."""
    if req.course_id not in _course_store:
        raise HTTPException(status_code=404, detail=f"Course not found: {req.course_id}")

    if req.course_id not in _embeddings:
        raise HTTPException(status_code=400, detail="No embedding available for this course")

    target_emb = _embeddings[req.course_id]
    results = []

    for cid, emb in _embeddings.items():
        if cid == req.course_id:
            continue
        course = _course_store.get(cid)
        if not course:
            continue
        if req.same_university and course.university != _course_store[req.course_id].university:
            continue

        # Cosine similarity
        sim = float(np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb) + 1e-10))
        results.append(CourseSearchResult(course=course, score=sim))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:req.limit]


@router.get("/stats/summary")
def course_stats():
    """Get course collection statistics."""
    courses = list(_course_store.values())
    universities = set(c.university for c in courses)
    levels = {}
    for c in courses:
        levels[c.level] = levels.get(c.level, 0) + 1

    return {
        "total_courses": len(courses),
        "universities": sorted(universities),
        "n_universities": len(universities),
        "courses_by_level": levels,
        "n_with_embeddings": len(_embeddings),
    }
