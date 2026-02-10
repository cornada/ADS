from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from ads_api.routes import (
    evaluate, pareto, telemetry, dashboard,
    courses, learner, assessment, temporal, transport, policy,
)

app = FastAPI(title="ADS API", version="0.2.0", description="Agent-Didactic Spaces API")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
app.include_router(pareto.router, prefix="/pareto", tags=["pareto"])
app.include_router(telemetry.router, prefix="/telemetry", tags=["telemetry"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(courses.router, prefix="/courses", tags=["courses"])
app.include_router(learner.router, prefix="/learner", tags=["learner"])
app.include_router(assessment.router, prefix="/assessment", tags=["assessment"])
app.include_router(temporal.router, prefix="/temporal", tags=["temporal"])
app.include_router(transport.router, prefix="/transport", tags=["transport"])
app.include_router(policy.router, prefix="/policy", tags=["policy"])

# Serve static UI files
UI_DIR = Path(__file__).parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    """Serve the main SPA."""
    app_file = UI_DIR / "app.html"
    if app_file.exists():
        return FileResponse(app_file)
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ADS API", "docs": "/docs"}


@app.get("/classic")
def classic_dashboard():
    """Serve the legacy Pareto-only dashboard."""
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Legacy dashboard not found"}
