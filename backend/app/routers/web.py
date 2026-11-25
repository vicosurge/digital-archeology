"""Server-rendered HTMX + Jinja2 UI for Digital Archeology.

Renders full pages and HTMX fragments, reusing the same services and models as the
JSON API. No build step, no separate frontend server.
"""
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Request, Depends, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, delete, text, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, defer

from ..database import get_db
from ..models import (
    Project, ArchivedFile, FileAnalysis, ArchivalCard, PageImage,
    ProjectStatus, FileStatus,
)
from ..services.wayback import wayback_service
from ..services.downloader import download_service
from ..services.llm import llm_service
from ..services import applog, enrichment, exclusions, local_view
from ..services.techstack import techstack_detector
from .projects import (
    sync_project_counts, reconcile_project_status, reset_failed_processing, apply_scope,
)
from ..config import get_llm_config, resolve_local_path

import asyncio
import logging

log = logging.getLogger("app.web")

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mimetypes offered as checkboxes on the filter form.
COMMON_MIMETYPES = [
    "text/html", "text/plain", "text/css", "application/pdf",
    "image/jpeg", "image/gif", "image/png", "application/javascript",
]

ACTIVE_STATUSES = {ProjectStatus.DOWNLOADING, ProjectStatus.PROCESSING}

router = APIRouter(tags=["web"])


# ----- Jinja filters -----
def _fmt_ts(ts: Optional[str]) -> str:
    """Format a Wayback YYYYMMDDHHMMSS timestamp as a readable date."""
    if not ts or len(ts) < 8 or not ts[:8].isdigit():
        return ts or ""
    out = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
    if len(ts) >= 12:
        out += f" {ts[8:10]}:{ts[10:12]}"
    return out


def _truncate(value: Optional[str], length: int = 70) -> str:
    if not value:
        return ""
    return value if len(value) <= length else value[: length - 1] + "…"


def _person_names(people) -> list:
    """Normalize a people_mentioned list (strings and/or dicts) to name strings."""
    return [n for n in (enrichment.person_to_str(p) for p in (people or [])) if n]


templates.env.filters["fmt_ts"] = _fmt_ts
templates.env.filters["truncate_url"] = _truncate
templates.env.filters["person_names"] = _person_names
templates.env.globals["local_href"] = local_view.local_href


# ----- helpers -----
async def _get_project(db: AsyncSession, project_id: int) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _is_active(project: Project) -> bool:
    return project.status in ACTIVE_STATUSES


# The workflow fragment polls every 2-3s; listing models on each poll would hit
# the (metered) endpoint constantly. Five minutes is plenty for a model list.
_MODELS_TTL = 300.0
_models_cache: dict = {"at": 0.0, "models": []}


async def _cached_models() -> List[str]:
    loop_now = asyncio.get_running_loop().time()
    if _models_cache["models"] and loop_now - _models_cache["at"] < _MODELS_TTL:
        return _models_cache["models"]
    status = await llm_service.check_ollama()
    models = status.get("models", []) if status.get("status") == "running" else []
    models = sorted(models)
    if models:
        _models_cache.update(at=loop_now, models=models)
    return models


async def _workflow_context(db: AsyncSession, project: Project) -> dict:
    await sync_project_counts(db, project)
    await reconcile_project_status(db, project)
    dl = await download_service.get_progress(project.id)
    proc = await llm_service.get_progress(project.id)
    # A run started by an earlier request (or before a restart) keeps the
    # fragment polling; reconcile_project_status is the backstop that stops it.
    active = _is_active(project) or proc.get("running", False)
    # Analysis counts cover only the in-scope corpus, so the step-4 numbers do
    # not add up against the download totals unless the exclusion is stated.
    excluded = 0
    if project.scope_start or project.scope_end or project.exclude_patterns:
        excluded = (await db.execute(
            select(func.count(ArchivedFile.id))
            .where(ArchivedFile.project_id == project.id)
            .where(ArchivedFile.in_scope.is_(False))
        )).scalar() or 0
    archive_error_pages = (await db.execute(
        select(func.count(ArchivedFile.id))
        .where(ArchivedFile.project_id == project.id)
        .where(ArchivedFile.status == FileStatus.SKIPPED)
        .where(ArchivedFile.error_message.like("Archive only holds an error page%")
               | ArchivedFile.error_message.like("Redirect target was never archived%"))
    )).scalar() or 0
    return {
        "project": project,
        "active": active,
        "archive_error_pages": archive_error_pages,
        "download": dl,
        "processing": proc,
        "scope_excluded": excluded,
        "suggested_exclusions": exclusions.SUGGESTED,
        "exclusion_count": len(exclusions.parse(project.exclude_patterns)),
        "mimetypes": COMMON_MIMETYPES,
        "ProjectStatus": ProjectStatus,
        "poll_interval": "3s" if proc.get("running") else "2s",
        # Only needed for the start form, which is hidden while a run is going.
        "models": [] if proc.get("running") else await _cached_models(),
    }


def _render(request: Request, name: str, context: dict) -> HTMLResponse:
    # Modern (request-first) Starlette signature — works across Starlette versions
    # and avoids the removed positional (name, context) form.
    return templates.TemplateResponse(request, name, context)


# ===== Logs =====
def _log_filters(level: str, q: str, project: str) -> dict:
    pid = int(project) if project.isdigit() else None
    return {"level": level if level in applog.LEVELS else "info", "q": q, "project": pid}


@router.get("/logs", response_class=HTMLResponse)
async def page_logs(request: Request, level: str = "info", q: str = "", project: str = "",
                    db: AsyncSession = Depends(get_db)):
    projects = (await db.execute(select(Project.id, Project.name).order_by(Project.id))).all()
    f = _log_filters(level, q, project)
    return _render(request, "logs.html", {
        "f": f, "projects": projects, "names": dict(projects), "records": applog.read(**f), "counts": applog.counts_since(24),
    })


@router.get("/_logs", response_class=HTMLResponse)
async def partial_logs(request: Request, level: str = "info", q: str = "", project: str = "",
                       db: AsyncSession = Depends(get_db)):
    names = dict((await db.execute(select(Project.id, Project.name))).all())
    f = _log_filters(level, q, project)
    return _render(request, "partials/_logs.html", {"records": applog.read(**f), "names": names})


# ===== Full pages =====
@router.get("/", response_class=HTMLResponse)
async def page_index(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    for p in projects:
        await sync_project_counts(db, p)
    ollama = await llm_service.check_ollama()
    models = ollama.get("models", []) if ollama.get("status") == "running" else []
    return _render(request, "project_list.html", {
        "projects": projects,
        "models": models,
        "ProjectStatus": ProjectStatus,
    })


@router.get("/project/{project_id}", response_class=HTMLResponse)
async def page_detail(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    ctx = await _workflow_context(db, project)
    return _render(request, "project_detail.html", ctx)


@router.get("/project/{project_id}/results", response_class=HTMLResponse)
async def page_results(project_id: int, request: Request, scope: str = Query("site"),
                       db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    if scope not in _SCOPE_CHOICES:
        scope = "site"
    stats = await _compute_stats(db, project_id, scope)
    return _render(request, "project_results.html", {
        "project": project,
        "stats": stats,
        "scope": scope,
        "scope_counts": await _scope_counts(db, project_id),
    })


@router.get("/project/{project_id}/card", response_class=HTMLResponse)
async def page_card(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    result = await db.execute(select(ArchivalCard).where(ArchivalCard.project_id == project_id))
    card = result.scalar_one_or_none()
    return _render(request, "archival_card.html", {"project": project, "card": card})


# ===== Project list actions =====
@router.post("/web/projects", response_class=HTMLResponse)
async def create_project(
    request: Request,
    name: str = Form(...),
    domain: str = Form(...),
    ollama_model: str = Form("llama3.2:3b"),
    db: AsyncSession = Depends(get_db),
):
    project = Project(name=name.strip(), domain=domain.strip(), ollama_model=ollama_model,
                      status=ProjectStatus.CREATED)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    # Send the user straight to the new project's workflow page.
    return Response(status_code=204, headers={"HX-Redirect": f"/project/{project.id}"})


@router.post("/project/{project_id}/delete", response_class=HTMLResponse)
async def delete_project(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    await db.delete(project)
    await db.commit()
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    for p in projects:
        await sync_project_counts(db, p)
    return _render(request, "partials/_project_grid.html", {
        "projects": projects, "ProjectStatus": ProjectStatus,
    })


# ===== Workflow fragment + actions =====
@router.get("/project/{project_id}/_workflow", response_class=HTMLResponse)
async def workflow_fragment(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    ctx = await _workflow_context(db, project)
    return _render(request, "partials/_workflow.html", ctx)


@router.post("/project/{project_id}/search-wayback", response_class=HTMLResponse)
async def action_search_wayback(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    project.status = ProjectStatus.FETCHING_URLS
    await db.commit()
    try:
        search = await wayback_service.search_domain(project.domain)
        project.total_snapshots = search["total_snapshots"]
        project.date_range_start = search["date_range_start"]
        project.date_range_end = search["date_range_end"]
        project.status = ProjectStatus.URLS_READY
        await db.commit()
    except Exception as e:
        log.warning("Wayback search failed: %s", e, extra={"project_id": project_id})
        project.status = ProjectStatus.ERROR
        await db.commit()
        ctx = await _workflow_context(db, project)
        ctx["error"] = f"Wayback search failed: {e}"
        return _render(request, "partials/_workflow.html", ctx)
    ctx = await _workflow_context(db, project)
    return _render(request, "partials/_workflow.html", ctx)


@router.post("/project/{project_id}/configure-filters", response_class=HTMLResponse)
async def action_configure_filters(
    project_id: int,
    request: Request,
    date_start: str = Form(""),
    date_end: str = Form(""),
    frequency: str = Form("all"),
    mimetypes: List[str] = Form([]),   # none ticked = every type
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project(db, project_id)
    if _is_active(project):
        return await _workflow_response(request, db, project_id,
                                        error="Stop the running download or analysis before re-queuing.")

    def to_wb(d: str) -> Optional[str]:
        d = (d or "").strip()
        return d.replace("-", "") if d else None

    start, end = to_wb(date_start), to_wb(date_end)
    if start and end and start > end:
        return await _workflow_response(request, db, project_id,
                                        error="The “From” date is after the “To” date.")

    try:
        snapshots = await wayback_service.get_all_snapshots(
            project.domain, date_start=start, date_end=end, mimetypes=mimetypes,
        )
    except Exception as e:
        log.warning("Capture list request failed: %s", e, extra={"project_id": project_id})
        return await _workflow_response(request, db, project_id,
                                        error=f"The Wayback Machine did not answer: {e}. Try again in a minute.")
    filtered = wayback_service.filter_by_frequency(snapshots, frequency)
    if not filtered:
        return await _workflow_response(request, db, project_id, error=(
            "No captures match these filters, so the queue was left as it was. "
            "Widen the dates or pick other content types."))

    project.filter_date_start, project.filter_date_end = start, end
    project.filter_frequency = frequency
    project.filter_mimetypes = mimetypes

    # Merge instead of replacing: captures already downloaded or analyzed are
    # kept (re-queuing used to delete them with their analyses). Only queued,
    # never-fetched captures the new filters no longer select are dropped.
    wanted = {(s["original_url"], s["timestamp"]): s for s in filtered}
    existing = (await db.execute(
        select(ArchivedFile.id, ArchivedFile.original_url, ArchivedFile.timestamp,
               ArchivedFile.status, ArchivedFile.local_path)
        .where(ArchivedFile.project_id == project_id)
    )).all()
    have, drop = set(), []
    for fid, url, ts, status, local_path in existing:
        key = (url, ts)
        if key in wanted:
            have.add(key)
        elif local_path is None and status in (FileStatus.PENDING, FileStatus.ERROR):
            drop.append(fid)
    for i in range(0, len(drop), 500):
        await db.execute(delete(ArchivedFile).where(ArchivedFile.id.in_(drop[i:i + 500])))
    added = 0
    for key, snap in wanted.items():
        if key in have:
            continue
        db.add(ArchivedFile(
            project_id=project_id,
            original_url=snap["original_url"],
            timestamp=snap["timestamp"],
            mimetype=snap["mimetype"],
            status_code=snap["status_code"],
            status=FileStatus.PENDING,
        ))
        added += 1
    await db.commit()
    project.files_total = (await db.execute(
        select(func.count(ArchivedFile.id)).where(ArchivedFile.project_id == project_id)
    )).scalar() or 0
    await db.commit()
    scope = await apply_scope(db, project)

    kept = len(existing) - len(drop)
    msg = f"{len(filtered)} captures match. {added} newly queued"
    if existing:
        msg += f", {kept} already in the project kept"
        if drop:
            msg += f", {len(drop)} queued ones no longer selected removed"
    msg += "."
    if scope["by_pattern"]:
        msg += f" {scope['by_pattern']} match the exclusion list and will not be fetched."
    return await _workflow_response(request, db, project_id, info=msg)


@router.post("/project/{project_id}/exclusions", response_class=HTMLResponse)
async def action_exclusions(
    project_id: int, request: Request,
    exclude_patterns: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project(db, project_id)
    if project.status == ProjectStatus.DOWNLOADING or (await llm_service.get_progress(project_id)).get("running"):
        return await _workflow_response(request, db, project_id,
                                        error="Stop the running download or analysis before changing exclusions.")
    project.exclude_patterns = exclude_patterns.strip() or None
    await db.commit()
    scope = await apply_scope(db, project)
    note = f"{scope['by_pattern']} captures match the exclusion list and are left out of the analysis."
    if scope["requeued"]:
        note += f" {scope['requeued']} no longer excluded are queued for download again."
    return await _workflow_response(request, db, project_id, info=note)


async def _workflow_response(request: Request, db: AsyncSession, project_id: int,
                             error: Optional[str] = None,
                             info: Optional[str] = None) -> HTMLResponse:
    project = await _get_project(db, project_id)
    ctx = await _workflow_context(db, project)
    if error:
        ctx["error"] = error
    if info:
        ctx["info"] = info
    return _render(request, "partials/_workflow.html", ctx)


@router.post("/project/{project_id}/start-download", response_class=HTMLResponse)
async def action_start_download(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    await _get_project(db, project_id)
    await download_service.start_download(project_id)
    return await _workflow_response(request, db, project_id)


@router.post("/project/{project_id}/stop-download", response_class=HTMLResponse)
async def action_stop_download(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    await download_service.stop_download(project_id)
    return await _workflow_response(request, db, project_id)


@router.post("/project/{project_id}/start-processing", response_class=HTMLResponse)
async def action_start_processing(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    concurrency: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
):
    await _get_project(db, project_id)
    result = await llm_service.start_processing(
        project_id, concurrency=concurrency, model=model or None
    )
    # start() refuses to run against an unreachable endpoint or a corpus that
    # is missing from disk; surface that as a notice rather than launching a
    # run that would mark every file as an error.
    error = result.get("message") if result.get("status") == "error" else None
    return await _workflow_response(
        request, db, project_id, error=error, info=result.get("note")
    )


@router.post("/project/{project_id}/stop-processing", response_class=HTMLResponse)
async def action_stop_processing(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    await llm_service.stop_processing(project_id)
    return await _workflow_response(request, db, project_id)


@router.post("/project/{project_id}/retry-failed-processing", response_class=HTMLResponse)
async def action_retry_failed_processing(
    project_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Requeue files that failed analysis. The JSON API has had this since the
    Vue days; the HTMX rewrite lost the button, leaving it unreachable."""
    await _get_project(db, project_id)
    result = await reset_failed_processing(db, project_id)
    return await _workflow_response(request, db, project_id, info=result["message"])


@router.get("/project/{project_id}/local")
async def local_capture(
    project_id: int, request: Request, url: str, ts: str = "",
    db: AsyncSession = Depends(get_db),
):
    """The downloaded copy of ``url`` nearest to ``ts``, readable without the Wayback Machine."""
    await _get_project(db, project_id)
    ts = ts if ts.isdigit() else ""
    hit = await local_view.resolve(db, project_id, url, ts)
    if not hit:
        # An image or stylesheet the page asked for: let the Archive supply it.
        if request.headers.get("sec-fetch-dest", "document") not in ("document", "iframe"):
            return RedirectResponse(f"https://web.archive.org/web/{ts or '2'}im_/{url}", status_code=302)
        return HTMLResponse(local_view.not_held_page(project_id, url, ts or "2"), status_code=404,
                            headers=local_view.SANDBOX_HEADERS)
    path, mimetype, capture_ts, kind = hit
    with open(path, "rb") as f:
        body = f.read()
    headers = dict(local_view.SANDBOX_HEADERS)
    if "html" in mimetype or (not mimetype and kind != "image"):
        body = local_view.rewrite_html(body, url, capture_ts, project_id)
        body = local_view.with_banner(body, local_view.banner(project_id, url, capture_ts, kind))
        charset = local_view.html_charset(body)
        media = "text/html" + (f"; charset={charset}" if charset else "")
    else:
        media = mimetype or "application/octet-stream"
    return Response(body, media_type=media, headers=headers)


@router.get("/project/{project_id}/_file-issues", response_class=HTMLResponse)
async def fragment_file_issues(project_id: int, request: Request,
                               db: AsyncSession = Depends(get_db)):
    """Every ERROR and SKIPPED file, grouped by reason, with the model's reply
    for failed analyses. Before this the only trace of a failure was a count."""
    await _get_project(db, project_id)
    rows = (await db.execute(
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .where(ArchivedFile.status.in_([FileStatus.ERROR, FileStatus.SKIPPED]))
        .order_by(ArchivedFile.error_message, ArchivedFile.original_url, ArchivedFile.timestamp)
    )).scalars().all()

    images: dict = {}
    skipped_ids = [r.id for r in rows if r.status == FileStatus.SKIPPED]
    if skipped_ids:
        for img in (await db.execute(
            select(PageImage).where(PageImage.file_id.in_(skipped_ids)).order_by(PageImage.id)
        )).scalars().all():
            images.setdefault(img.file_id, []).append(img)

    groups = []
    for status, label, note in (
        (FileStatus.ERROR, "Failed",
         "Retryable. “Retry failed” requeues these."),
        (FileStatus.SKIPPED, "Skipped",
         "Not sent to the model: nothing to analyze, or the archive only holds an error page."),
    ):
        mine = [r for r in rows if r.status == status]
        if not mine:
            continue
        by_reason: dict = {}
        for r in mine:
            by_reason.setdefault(r.error_message or "", []).append(r)
        groups.append({
            "label": label, "note": note, "rows": mine,
            "by_reason": sorted(by_reason.items(), key=lambda kv: -len(kv[1])),
        })
    return _render(request, "partials/_file_issues.html", {"groups": groups, "images": images})


# ===== Results fragment =====
# Analyses that were copied to duplicate captures carry source_file_id. They must
# be excluded from aggregates or every count is inflated by the duplicate rate
# (~1.5x on this corpus). Individual result rows still show them - see _results.html.
_ORIGINALS_ONLY = "fa.source_file_id IS NULL"

# Which era of the domain's capture history a view is about (see
# Project.scope_start). "site" is the default everywhere: a project with no
# scope window has every capture in scope, so this is a no-op until a window is
# set, and once one is set the pages describe the site rather than whatever took
# the domain over afterwards. "excluded" exists so the excluded era stays
# reachable and visible -- it is evidence, not something to hide.
_SCOPE_CHOICES = ("site", "excluded", "all")


def _scope_sql(scope: str) -> str:
    if scope == "excluded":
        return "af.in_scope = 0"
    if scope == "all":
        return "1 = 1"
    return "af.in_scope = 1"

# Person entries may be plain strings or objects like {"name": ...}; normalize in SQL
# so filtering and aggregation agree with the _person_names Jinja filter.
_PERSON_VALUE = (
    "CASE WHEN json_valid(je.value) AND json_type(je.value) = 'object' "
    "THEN json_extract(je.value, '$.name') ELSE je.value END"
)


async def _scope_counts(db: AsyncSession, project_id: int) -> dict:
    """Analyzed-page counts on each side of the scope boundary.

    Drives the scope toggle's labels, and tells the template whether an
    excluded era exists at all.
    """
    rows = (await db.execute(
        text("SELECT af.in_scope, COUNT(*) "
             "FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id "
             "WHERE af.project_id = :pid AND af.status = 'PROCESSED' "
             "GROUP BY af.in_scope"),
        {"pid": project_id},
    )).fetchall()
    counts = {int(flag): int(n) for flag, n in rows}
    return {
        "site": counts.get(1, 0),
        "excluded": counts.get(0, 0),
        "all": sum(counts.values()),
    }


async def _compute_stats(db: AsyncSession, project_id: int, scope: str = "site") -> dict:
    """Aggregate the results overview.

    Done as grouped queries rather than by materializing every analysis: at
    tens of thousands of rows the previous approach loaded the whole table
    (including raw_response) on each results-page view.
    """
    scope_sql = _scope_sql(scope)
    base = (
        "FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id "
        f"WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND {scope_sql}"
    )
    params = {"pid": project_id}

    processed = (await db.execute(
        text(f"SELECT COUNT(*) {base}"), params
    )).scalar() or 0

    async def _group(column: str) -> dict:
        rows = (await db.execute(
            text(f"SELECT {column} AS k, COUNT(*) {base} AND {_ORIGINALS_ONLY} "
                 f"AND {column} IS NOT NULL AND {column} != '' GROUP BY k ORDER BY COUNT(*) DESC"),
            params,
        )).fetchall()
        return {k: int(n) for k, n in rows}

    sentiments = await _group("fa.sentiment")
    content_types = await _group("fa.content_type")

    tag_rows = (await db.execute(
        text("SELECT je.value AS tag, COUNT(*) AS n "
             "FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id, "
             "json_each(fa.tags) je "
             f"WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND {scope_sql} "
             f"AND {_ORIGINALS_ONLY} "
             "AND je.value IS NOT NULL AND je.value != '' "
             "GROUP BY tag ORDER BY n DESC LIMIT 20"),
        params,
    )).fetchall()

    people_rows = (await db.execute(
        text(f"SELECT {_PERSON_VALUE} AS person, COUNT(*) AS n "
             "FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id, "
             "json_each(fa.people_mentioned) je "
             f"WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND {scope_sql} "
             f"AND {_ORIGINALS_ONLY} "
             "AND person IS NOT NULL AND person != '' "
             "GROUP BY person ORDER BY n DESC LIMIT 40"),
        params,
    )).fetchall()

    return {
        "processed": processed,
        "sentiments": sentiments,
        "content_types": content_types,
        "top_tags": [(t, int(n)) for t, n in tag_rows],
        "people": [p for p, _ in people_rows],
    }


@router.get("/project/{project_id}/_results", response_class=HTMLResponse)
async def results_fragment(
    project_id: int,
    request: Request,
    q: str = Query(""),
    sentiment: str = Query(""),
    content_type: str = Query(""),
    person: str = Query(""),
    tag: str = Query(""),
    scope: str = Query("site"),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
):
    per_page = 20
    if scope not in _SCOPE_CHOICES:
        scope = "site"

    # Filter and paginate in SQL. Doing this in Python meant loading every
    # analysed row (with its raw_response) to render 20 of them.
    clauses = ["af.project_id = :pid", "af.status = 'PROCESSED'", _scope_sql(scope)]
    params = {"pid": project_id}
    if q:
        clauses.append("LOWER(fa.description) LIKE :q")
        params["q"] = f"%{q.lower()}%"
    if sentiment:
        clauses.append("fa.sentiment = :sentiment")
        params["sentiment"] = sentiment
    if content_type:
        clauses.append("fa.content_type = :content_type")
        params["content_type"] = content_type
    if tag:
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(fa.tags) je WHERE LOWER(je.value) = :tag)"
        )
        params["tag"] = tag.lower()
    if person:
        clauses.append(
            f"EXISTS (SELECT 1 FROM json_each(fa.people_mentioned) je "
            f"WHERE LOWER({_PERSON_VALUE}) = :person)"
        )
        params["person"] = person.lower()

    where = " AND ".join(clauses)
    base = f"FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id WHERE {where}"

    total = (await db.execute(text(f"SELECT COUNT(*) {base}"), params)).scalar() or 0
    pages = (total + per_page - 1) // per_page

    id_rows = (await db.execute(
        text(f"SELECT af.id {base} ORDER BY af.timestamp DESC, af.id LIMIT :limit OFFSET :offset"),
        {**params, "limit": per_page, "offset": (page - 1) * per_page},
    )).fetchall()
    page_ids = [r[0] for r in id_rows]

    page_files = []
    if page_ids:
        rows = await db.execute(
            select(ArchivedFile)
            .where(ArchivedFile.id.in_(page_ids))
            # raw_response holds the full model output and is never rendered here.
            .options(selectinload(ArchivedFile.analysis).defer(FileAnalysis.raw_response))
        )
        by_id = {f.id: f for f in rows.scalars().all()}
        page_files = [by_id[i] for i in page_ids if i in by_id]

    return _render(request, "partials/_results.html", {
        "project_id": project_id,
        "files": page_files,
        "total": total,
        "page": page,
        "pages": pages,
        "scope": scope,
        "scope_counts": await _scope_counts(db, project_id),
        "filters": {"q": q, "sentiment": sentiment, "content_type": content_type,
                    "person": person, "tag": tag},
    })


# ===== Archival card =====
async def _build_card(db: AsyncSession, project: Project) -> ArchivalCard:
    """Run all enrichment sources and upsert the project's archival card."""
    # Gather inputs from already-stored data. Each query pulls only what it
    # feeds: loading every ArchivedFile with every analysis attached made this
    # scale with the corpus even though the card only needs a sample.
    #
    # Every input is restricted to in-scope captures. The card is a profile of
    # one site, and on a domain that changed hands the excluded era would
    # otherwise dominate all three inputs: the timeline would show the site's
    # busiest years long after it died, tech detection samples the *newest* HTML
    # and so would fingerprint the squatter's stack, and the people and summary
    # would be drawn from generated spam.
    timestamps = [
        r[0] for r in (await db.execute(
            select(ArchivedFile.timestamp)
            .where(ArchivedFile.project_id == project.id)
            .where(ArchivedFile.in_scope.is_(True))
            .where(ArchivedFile.timestamp.isnot(None))
        )).all()
    ]

    # techstack_detector samples 40 files internally; give it a modest candidate
    # pool of HTML rather than every path in the project.
    path_rows = (await db.execute(
        select(ArchivedFile.local_path)
        .where(ArchivedFile.project_id == project.id)
        .where(ArchivedFile.in_scope.is_(True))
        .where(ArchivedFile.local_path.isnot(None))
        .where(ArchivedFile.mimetype.like("text/html%"))
        .order_by(ArchivedFile.timestamp.desc())
        .limit(200)
    )).all()
    local_paths = [p for p in (resolve_local_path(r[0]) for r in path_rows) if p]

    # Only directly-analyzed rows: copies would skew the people counts.
    analyses = (await db.execute(
        select(FileAnalysis)
        .join(ArchivedFile, ArchivedFile.id == FileAnalysis.file_id)
        .where(ArchivedFile.project_id == project.id)
        .where(ArchivedFile.in_scope.is_(True))
        .where(FileAnalysis.source_file_id.is_(None))
        .options(defer(FileAnalysis.raw_response))
        .limit(400)
    )).scalars().all()

    # 1. Free ownership lookup (RDAP)
    whois = await enrichment.fetch_rdap(project.domain)
    # 2. Timeline from capture timestamps
    timeline = enrichment.build_timeline(timestamps)
    # 3. Offline tech-stack detection over archived HTML (CPU-bound -> thread)
    tech_stack = await asyncio.to_thread(techstack_detector.detect_in_files, local_paths)
    # 4. People surfaced from content analysis
    people = enrichment.aggregate_people(analyses)
    # 5. LLM narrative synthesis
    synth = await llm_service.synthesize_domain_card(
        project.domain, project.ollama_model, analyses, timeline, tech_stack, whois, people,
    )

    # Upsert card
    existing = await db.execute(select(ArchivalCard).where(ArchivalCard.project_id == project.id))
    card = existing.scalar_one_or_none()
    if not card:
        card = ArchivalCard(project_id=project.id)
        db.add(card)

    card.registrar = whois["registrar"]
    card.whois_created = whois["whois_created"]
    card.whois_expires = whois["whois_expires"]
    card.whois_updated = whois["whois_updated"]
    card.nameservers = whois["nameservers"]
    card.whois_status = whois["whois_status"]
    card.registrant = whois["registrant"]
    card.whois_raw = whois["whois_raw"]
    card.whois_source = whois["whois_source"]
    card.whois_domain = whois["whois_domain"]
    card.tech_stack = tech_stack
    card.timeline = timeline
    card.likely_people = people
    card.llm_summary = synth["summary"]
    card.llm_model_used = synth["model"]
    card.generated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(card)
    return card


@router.post("/project/{project_id}/build-card", response_class=HTMLResponse)
async def action_build_card(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    card = await _build_card(db, project)
    return _render(request, "partials/_card.html", {"project": project, "card": card})


# ===== Ollama status fragment (live header indicator) =====
@router.get("/_ollama-status", response_class=HTMLResponse)
async def ollama_status_fragment(request: Request):
    status = await llm_service.check_ollama()
    return _render(request, "partials/_ollama_status.html", {"ollama": status})
