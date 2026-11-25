"""Digital Archeology - FastAPI Application"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

import logging

from .config import get_data_dir, get_llm_config
from .services import applog

applog.setup("web")
log = logging.getLogger("app")
from .database import init_db
from .routers import projects, export, web, research
from .services.llm import llm_service
from .services.processing import processing_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await init_db()
    log.info("App started; database at %s", get_data_dir())

    # Clear rows and runs stranded by a process that died mid-analysis. Does not
    # auto-resume: the UI offers a Resume button instead, so a restart never
    # silently starts spending against a metered endpoint.
    await processing_worker.recover_orphans()

    # Check Ollama. The host is printed; the API key never is.
    cfg = get_llm_config()
    ollama_status = await llm_service.check_ollama(cfg)
    if ollama_status["status"] == "running":
        log.info("Ollama reachable at %s (auth: %s), %d models",
                 cfg.safe_base_url(), "yes" if cfg.has_auth else "no", len(ollama_status["models"]))
    else:
        log.warning("Ollama not reachable at %s: %s",
                    cfg.safe_base_url(), ollama_status.get("message", ""))

    yield

    # Shutdown
    log.info("Shutting down")


app = FastAPI(
    title="Digital Archeology",
    description="Recover and analyze archived websites from the Wayback Machine",
    version="1.0.0",
    lifespan=lifespan,
)

# The UI is server-rendered and same-origin, so no CORS middleware is needed.
# (The previous config paired allow_origins=["*"] with allow_credentials=True,
# which browsers reject anyway; it existed for the since-removed Vue SPA.)

# Static assets (htmx, css) for the server-rendered UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(projects.router, prefix="/api")
app.include_router(export.router, prefix="/api")
# Server-rendered HTMX UI (mounted at root)
app.include_router(web.router)
app.include_router(research.router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = await llm_service.check_ollama()
    return {
        "status": "ok",
        "ollama": ollama_status,
    }


@app.get("/api/ollama/models")
async def get_ollama_models():
    """Get available Ollama models"""
    status = await llm_service.check_ollama()
    if status["status"] == "running":
        return {"models": status["models"]}
    return {"error": status.get("message", "Ollama not running"), "models": []}
