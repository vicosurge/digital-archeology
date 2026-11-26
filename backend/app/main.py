"""Digital Archeology - FastAPI Application"""
import base64
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import os

import logging

from .config import get_auth, get_data_dir, get_llm_config
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

# Optional password (DA_PASSWORD). HTTP Basic: the browser asks once and then
# sends it with every request, HTMX's included, so no login page is needed.
# The health check stays open for Docker's healthcheck.
_OPEN_PATHS = {"/api/health"}


def _authorized(header: str, user: str, password: str) -> bool:
    if not header.lower().startswith("basic "):
        return False
    try:
        given_user, _, given_pw = base64.b64decode(header[6:].strip()).decode("utf-8").partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    ok_user = secrets.compare_digest(given_user.encode(), user.encode())
    ok_pw = secrets.compare_digest(given_pw.encode(), password.encode())
    return ok_user and ok_pw


@app.middleware("http")
async def require_password(request: Request, call_next):
    creds = get_auth()
    if creds is None or request.url.path in _OPEN_PATHS:
        return await call_next(request)
    if _authorized(request.headers.get("authorization", ""), *creds):
        return await call_next(request)
    return Response("Password required.", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Digital Archeology", charset="UTF-8"'})


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
