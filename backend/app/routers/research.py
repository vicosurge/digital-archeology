"""The research workflow page: the method, one step after another.

Scope -> themes over time -> follow the links -> findings -> the historian's
archive. Long jobs (link recovery, link analysis) run as background tasks and
the status fragment polls while any is running.
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import ArchivalCard, Project, ProjectStatus
from ..services import apparatus, archive_export, estimate, linked_sites, memento, takeover, warc_export
from ..services.findings import build_findings
from ..services.links import link_service
from ..services.reading import GENRES, INTENTS, STANCES, reading_service
from ..services.themes import compute_themes, sentiment_shift
from .web import _get_project, _is_active, _render

log = logging.getLogger("app.research")

router = APIRouter(tags=["research"])

LINK_KIND_ORDER = ["reference", "article", "document", "sitewide", "social", "newsletter",
                   "advertising", "navigation", "taxonomy", "media", "download", "other"]


async def _estimate_line(db: AsyncSession, project_id: int, job: str) -> str:
    project = await db.get(Project, project_id)
    model = project.ollama_model if project else None
    return estimate.describe(await estimate.estimate(db, project_id, job, model))


async def _links_context(db: AsyncSession, project_id: int) -> dict:
    rows = (await db.execute(text("""
        SELECT is_internal, kind, recovery_status, analysis_status, COUNT(*)
        FROM link_targets WHERE project_id = :pid GROUP BY 1, 2, 3, 4
    """), {"pid": project_id})).fetchall()
    kinds: dict = {}
    recovery: dict = {}
    analysis: dict = {}
    total = 0
    for is_internal, kind, rstatus, astatus, n in rows:
        key = ("internal" if is_internal else "external", kind)
        kinds[key] = kinds.get(key, 0) + n
        recovery[rstatus] = recovery.get(rstatus, 0) + n
        analysis[astatus] = analysis.get(astatus, 0) + n
        total += n
    citations = (await db.execute(text(
        "SELECT COUNT(*) FROM page_links WHERE project_id = :pid"), {"pid": project_id})).scalar() or 0
    unconfirmed = (await db.execute(text(
        "SELECT COUNT(*) FROM link_targets WHERE project_id = :pid AND recovery_status = 'not_archived' "
        "AND error_message LIKE 'unconfirmed%'"), {"pid": project_id})).scalar() or 0

    jobs = {k: link_service.job(project_id, k) for k in ("extract", "recover", "analyze")}
    jobs["memento"] = memento.job(project_id)
    by_archive = dict((await db.execute(text(
        "SELECT source_archive, COUNT(*) FROM link_targets WHERE project_id = :pid "
        "AND recovery_status = 'recovered' AND source_archive IS NOT NULL GROUP BY 1"),
        {"pid": project_id})).fetchall())
    to_recover = recovery.get("pending", 0)
    done_recover = recovery.get("recovered", 0) + recovery.get("not_archived", 0) + recovery.get("error", 0)
    to_analyze = analysis.get("pending", 0)
    done_analyze = analysis.get("processed", 0) + analysis.get("error", 0) + analysis.get("skipped", 0)
    return {
        "total_links": total, "citations": citations,
        "kinds": sorted(kinds.items(), key=lambda kv: (kv[0][0], LINK_KIND_ORDER.index(kv[0][1])
                                                        if kv[0][1] in LINK_KIND_ORDER else 99)),
        "recovery": recovery, "analysis": analysis, "unconfirmed": unconfirmed,
        "memento_pending": await memento.pending(db, project_id), "by_archive": by_archive,
        "recover_pct": int(100 * done_recover / (done_recover + to_recover)) if (done_recover + to_recover) else 0,
        "analyze_pct": int(100 * done_analyze / (done_analyze + to_analyze)) if (done_analyze + to_analyze) else 0,
        "jobs": jobs, "any_running": any(j.running for j in jobs.values()),
        "estimate": await _estimate_line(db, project_id, "links") if to_analyze else "",
    }


async def _card(db: AsyncSession, project_id: int) -> Optional[ArchivalCard]:
    return (await db.execute(select(ArchivalCard).where(ArchivalCard.project_id == project_id))).scalar_one_or_none()


def _file_info(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    stat = os.stat(path)
    return {"name": os.path.basename(path), "size_mb": round(stat.st_size / 1048576, 1),
            "built_at": datetime.fromtimestamp(stat.st_mtime)}


def _export_context(project) -> dict:
    return {"export": _file_info(archive_export.latest_export(project)),
            "warc": _file_info(warc_export.latest_warc(project))}


async def _reading_context(project_id: int, db: Optional[AsyncSession] = None) -> dict:
    stats = await reading_service.stats(project_id)
    line = await _estimate_line(db, project_id, "readings") if db is not None else ""
    return {**stats, "job": reading_service.job(project_id), "estimate": line}


@router.get("/project/{project_id}/research", response_class=HTMLResponse)
async def page_research(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    themes = await compute_themes(db, project_id)
    card = await _card(db, project_id)
    return _render(request, "research.html", {
        "project": project, "themes": themes, "shift": sentiment_shift(themes["years"]),
        "card": card, "links": await _links_context(db, project_id),
        "reading": await _reading_context(project_id, db), **_export_context(project),
    })


# ----- the period under study: takeover detection and scope -----
def _ts(value: str) -> Optional[str]:
    """'2017-06-17' or a 14-digit timestamp -> 14 digits, or None."""
    digits = "".join(c for c in (value or "") if c.isdigit())
    if not digits:
        return None
    if len(digits) < 8:
        raise HTTPException(status_code=400, detail="Dates need at least YYYY-MM-DD")
    return digits[:14].ljust(14, "0")


async def _scope_context(db: AsyncSession, project: Project, message: str = "") -> dict:
    result = await takeover.detect_cached(db, project.id)
    proposal = result["proposal"]
    # Any end between the last good capture and the proposed one selects the
    # same captures, so it agrees with the proposal.
    applied = bool(proposal and project.scope_end
                   and (proposal.get("last_good") or "") < project.scope_end <= proposal["scope_end"])
    return {"project": project, "years": result["years"], "proposal": proposal,
            "applied": applied, "message": message}


@router.get("/project/{project_id}/_takeover", response_class=HTMLResponse)
async def takeover_fragment(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    return _render(request, "partials/_takeover.html", await _scope_context(db, project))


@router.post("/project/{project_id}/scope", response_class=HTMLResponse)
async def scope_action(project_id: int, request: Request, action: str = Form("apply"),
                       scope_start: str = Form(""), scope_end: str = Form(""),
                       scope_reason: str = Form(""), db: AsyncSession = Depends(get_db)):
    from .projects import apply_scope
    from ..services.llm import llm_service
    project = await _get_project(db, project_id)
    running = _is_active(project) or (await llm_service.get_progress(project_id)).get("running")
    if running:
        return _render(request, "partials/_takeover.html", await _scope_context(
            db, project, "A download or analysis is running. Stop it before changing the scope."))
    if action == "clear":
        project.scope_start = project.scope_end = project.scope_reason = None
    else:
        start, end = _ts(scope_start), _ts(scope_end)
        if start and end and start >= end:
            raise HTTPException(status_code=400, detail="The start must come before the end")
        project.scope_start, project.scope_end = start, end
        project.scope_reason = scope_reason.strip() or None
    counts = await apply_scope(db, project)
    message = (f"Scope cleared: all {counts['in_scope']:,} captures are in scope." if action == "clear" else
               f"Scope applied: {counts['in_scope']:,} captures in scope, {counts['excluded']:,} kept aside.")
    return _render(request, "partials/_takeover.html", await _scope_context(db, project, message))


# ----- following the links to other sites -----
@router.get("/project/{project_id}/_linked-sites", response_class=HTMLResponse)
async def linked_sites_fragment(project_id: int, request: Request, platforms: bool = Query(False),
                                db: AsyncSession = Depends(get_db)):
    await _get_project(db, project_id)
    sites = await linked_sites.summary(db, project_id, limit=500)
    shown = [s for s in sites if platforms or not s["platform"]][:60]
    return _render(request, "partials/_linked_sites.html", {
        "project_id": project_id, "sites": shown, "platforms": platforms,
        "hidden_platforms": sum(1 for s in sites if s["platform"]) if not platforms else 0,
    })


@router.post("/project/{project_id}/follow", response_class=HTMLResponse)
async def follow_site(project_id: int, site: str = Form(...), db: AsyncSession = Depends(get_db)):
    """Start a project on a linked site, or open the one that exists."""
    source = await _get_project(db, project_id)
    site = linked_sites.site_of(site)
    if not site or "." not in site:
        raise HTTPException(status_code=400, detail="Not a site")
    existing = await linked_sites.existing_project(db, site)
    if existing:
        return Response(status_code=204, headers={"HX-Redirect": f"/project/{existing}"})
    project = Project(name=site, domain=site, ollama_model=source.ollama_model,
                      status=ProjectStatus.CREATED, source_project_id=source.id)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    log.info("Started project %d on %s from the links of project %d", project.id, site, source.id,
             extra={"project_id": project.id})
    return Response(status_code=204, headers={"HX-Redirect": f"/project/{project.id}"})


# ----- close reading -----
@router.get("/project/{project_id}/_reading-status", response_class=HTMLResponse)
async def reading_status(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    return _render(request, "partials/_reading_status.html",
                   {"project": project, "reading": await _reading_context(project_id, db)})


@router.post("/project/{project_id}/reading/{action}", response_class=HTMLResponse)
async def reading_action(project_id: int, action: str, request: Request,
                         limit: int = Form(0), db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    info = None
    if action == "select":
        result = await reading_service.select_pages(project_id)
        info = f"Queued {result['queued']} article page{'s' if result['queued'] != 1 else ''}."
    elif action == "run":
        reading_service.start_read(project_id, limit=max(0, limit))
    elif action == "stop":
        reading_service.stop(project_id)
    elif action in ("requeue-errors", "requeue-all"):
        n = await reading_service.requeue(project_id, action.split("-", 1)[1])
        info = f"Queued {n} page{'s' if n != 1 else ''} to be read again."
    else:
        raise HTTPException(status_code=404, detail="Unknown action")
    return _render(request, "partials/_reading_status.html",
                   {"project": project, "reading": await _reading_context(project_id, db), "info": info})


@router.get("/project/{project_id}/_apparatus", response_class=HTMLResponse)
async def apparatus_fragment(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    return _render(request, "partials/_apparatus.html",
                   {"project": project, "app": await apparatus.build(db, project_id)})


@router.get("/project/{project_id}/_readings-table", response_class=HTMLResponse)
async def readings_table(project_id: int, request: Request,
                         genre: str = Query(""), intent: str = Query(""), stance: str = Query(""),
                         q: str = Query(""), page: int = Query(1, ge=1),
                         db: AsyncSession = Depends(get_db)):
    per_page = 20
    clauses = ["pr.project_id = :pid", "pr.status = 'processed'", "pr.source_file_id IS NULL"]
    params: dict = {"pid": project_id}
    for field, value in (("genre", genre), ("intent", intent), ("stance", stance)):
        if value:
            clauses.append(f"pr.{field} = :{field}")
            params[field] = value
    if q:
        clauses.append("(LOWER(COALESCE(pr.title,'')) LIKE :q OR LOWER(pr.summary) LIKE :q "
                       "OR LOWER(COALESCE(pr.people,'')) LIKE :q OR LOWER(COALESCE(pr.places,'')) LIKE :q)")
        params["q"] = f"%{q.lower()}%"
    where = " AND ".join(clauses)
    base = ("FROM page_readings pr JOIN archived_files af ON af.id = pr.file_id "
            f"WHERE {where}")
    total = (await db.execute(text(f"SELECT COUNT(*) {base}"), params)).scalar() or 0
    rows = (await db.execute(text(f"""
        SELECT pr.file_id, pr.title, pr.genre, pr.intent, pr.stance, pr.summary,
               pr.people, pr.places, pr.events, pr.facts, pr.quotes, pr.frames,
               pr.published_estimate, pr.period_start, pr.period_end, pr.temporal_mode,
               af.original_url, af.timestamp
        {base} ORDER BY af.timestamp, pr.file_id LIMIT :limit OFFSET :offset
    """), {**params, "limit": per_page, "offset": (page - 1) * per_page})).mappings().all()
    # Raw SQL hands back JSON columns as text; the template wants lists.
    decoded = []
    for row in rows:
        record = dict(row)
        for field in ("people", "places", "events", "facts", "quotes", "frames"):
            try:
                record[field] = json.loads(record[field]) if record[field] else []
            except (TypeError, ValueError):
                record[field] = []
        decoded.append(record)
    return _render(request, "partials/_readings_table.html", {
        "project_id": project_id, "rows": decoded, "total": total, "page": page,
        "pages": (total + per_page - 1) // per_page,
        "filters": {"genre": genre, "intent": intent, "stance": stance, "q": q},
        "genres": sorted(GENRES), "intents": sorted(INTENTS), "stances": sorted(STANCES),
    })


# ----- links -----
@router.get("/project/{project_id}/_links-status", response_class=HTMLResponse)
async def links_status(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    return _render(request, "partials/_links_status.html",
                   {"project": project, "links": await _links_context(db, project_id)})


@router.post("/project/{project_id}/links/{action}", response_class=HTMLResponse)
async def links_action(project_id: int, action: str, request: Request,
                       limit: int = Form(0), db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    info = None
    if action == "extract":
        link_service.start_extract(project_id)
    elif action == "recover":
        link_service.start_recover(project_id, limit=max(0, limit))
    elif action == "analyze":
        link_service.start_analyze(project_id, limit=max(0, limit))
    elif action == "memento":
        memento.start_links(project_id, limit=max(0, limit))
    elif action == "stop-memento":
        memento.stop_links(project_id)
    elif action in ("stop-extract", "stop-recover", "stop-analyze"):
        link_service.stop(project_id, action.split("-", 1)[1])
    elif action in ("requeue-unconfirmed", "requeue-errors", "requeue-analysis_errors"):
        n = await link_service.requeue(project_id, action.split("-", 1)[1])
        info = f"Queued {n} link{'s' if n != 1 else ''} again."
    else:
        raise HTTPException(status_code=404, detail="Unknown action")
    ctx = {"project": project, "links": await _links_context(db, project_id), "info": info}
    return _render(request, "partials/_links_status.html", ctx)


@router.get("/project/{project_id}/_links-table", response_class=HTMLResponse)
async def links_table(project_id: int, request: Request,
                      scope: str = Query("external"), kind: str = Query(""),
                      status: str = Query("recovered"), q: str = Query(""),
                      page: int = Query(1, ge=1), db: AsyncSession = Depends(get_db)):
    per_page = 25
    clauses = ["project_id = :pid"]
    params: dict = {"pid": project_id}
    if scope in ("external", "internal"):
        clauses.append("is_internal = :internal")
        params["internal"] = 1 if scope == "internal" else 0
    if kind:
        clauses.append("kind = :kind")
        params["kind"] = kind
    if status:
        clauses.append("recovery_status = :status")
        params["status"] = status
    if q:
        clauses.append("(LOWER(url) LIKE :q OR LOWER(COALESCE(title, '')) LIKE :q "
                       "OR LOWER(COALESCE(description, '')) LIKE :q OR LOWER(COALESCE(sample_anchor, '')) LIKE :q)")
        params["q"] = f"%{q.lower()}%"
    where = " AND ".join(clauses)
    total = (await db.execute(text(f"SELECT COUNT(*) FROM link_targets WHERE {where}"), params)).scalar() or 0
    rows = (await db.execute(text(f"""
        SELECT id, url, domain, kind, cited_by_pages, first_cited_ts, wayback_timestamp, wayback_url,
               capture_distance_days, recovery_status, analysis_status, title, description,
               relation, relation_note, sample_anchor, error_message, source_archive
        FROM link_targets WHERE {where}
        ORDER BY cited_by_pages DESC, id LIMIT :limit OFFSET :offset
    """), {**params, "limit": per_page, "offset": (page - 1) * per_page})).mappings().all()
    return _render(request, "partials/_links_table.html", {
        "project_id": project_id, "rows": rows, "total": total, "page": page,
        "pages": (total + per_page - 1) // per_page,
        "filters": {"scope": scope, "kind": kind, "status": status, "q": q},
        "kinds": LINK_KIND_ORDER,
    })


# ----- findings -----
@router.post("/project/{project_id}/build-findings", response_class=HTMLResponse)
async def action_build_findings(project_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    card = await _card(db, project_id)
    error = None
    try:
        findings, model = await build_findings(db, project)
        if not card:
            card = ArchivalCard(project_id=project_id)
            db.add(card)
        card.findings = findings
        card.findings_model = model
        card.findings_generated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(card)
    except Exception as exc:  # surface in the page; the endpoint or the model may be at fault
        from ..config import get_llm_config
        error = get_llm_config().redact(f"{type(exc).__name__}: {exc}")
    return _render(request, "partials/_findings.html", {"project": project, "card": card, "error": error})


# ----- export -----
@router.post("/project/{project_id}/export", response_class=HTMLResponse)
async def action_export(project_id: int, request: Request, include_captures: str = Form("1"),
                        db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    error = None
    try:
        await archive_export.build_archive(db, project, include_captures=include_captures == "1")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _render(request, "partials/_export.html", {"project": project, "error": error,
                                                      **_export_context(project)})


@router.post("/project/{project_id}/warc", response_class=HTMLResponse)
async def action_warc(project_id: int, request: Request, all_captures: str = Form(""),
                      db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    error = info = None
    try:
        r = await warc_export.build_warc(db, project, all_captures=all_captures == "1")
        info = f"WARC written: {r['resource']:,} captures and {r['metadata']:,} analyses."
        if r["missing"]:
            info += f" {r['missing']:,} files were missing on disk and left out."
        log.info(info, extra={"project_id": project_id})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _render(request, "partials/_export.html", {"project": project, "error": error, "info": info,
                                                      **_export_context(project)})


@router.get("/project/{project_id}/warc/download")
async def warc_download(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    path = warc_export.latest_warc(project)
    if not path:
        raise HTTPException(status_code=404, detail="No WARC has been built yet")
    return FileResponse(path, media_type="application/warc", filename=os.path.basename(path))


@router.get("/project/{project_id}/export/download")
async def export_download(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await _get_project(db, project_id)
    path = archive_export.latest_export(project)
    if not path:
        raise HTTPException(status_code=404, detail="No archive has been built yet")
    return FileResponse(path, media_type="application/zip", filename=os.path.basename(path))
