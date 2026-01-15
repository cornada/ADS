from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from ads_api.routes import evaluate, pareto, telemetry, dashboard

app = FastAPI(title="ADS API", version="0.1.0", description="Agent-Didactic Spaces API")

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

# Serve static UI files
UI_DIR = Path(__file__).parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    """Serve the dashboard UI."""
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ADS API", "docs": "/docs", "dashboard": "/static/index.html"}
