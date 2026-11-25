"""API routes for project management"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, update, and_, or_, true
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models import (
    Project, ArchivedFile, FileAnalysis, ProjectStatus, FileStatus,
    ProjectCreate, ProjectResponse, ProjectFilterSettings, ProjectProgress,
    FileResponse, AnalysisResponse, FileWithAnalysis, SearchQuery, SearchResults
)
from ..services.wayback import wayback_service
from ..services.downloader import download_service
from ..services import exclusions
from ..services.llm import llm_service

router = APIRouter(prefix="/projects", tags=["projects"])


async def count_by_status(db: AsyncSession, project_id: int) -> dict:
    """File counts keyed by FileStatus name, in one grouped query.

    This runs for every project on the index page and on every 2s workflow
    poll, so it replaces the three separate COUNT(*) queries it grew from.
    """
    rows = await db.execute(
        select(ArchivedFile.status, func.count(ArchivedFile.id))
        .where(ArchivedFile.project_id == project_id)
        .group_by(ArchivedFile.status)
    )
    counts: dict = {}
    for status, count in rows.all():
        key = status.name if hasattr(status, "name") else str(status)
        counts[key] = int(count)
    return counts


async def apply_scope(db: AsyncSession, project: Project) -> dict:
    """Recompute ArchivedFile.in_scope from the project's scope window.

    Idempotent, and safe to run mid-corpus: it only flips the flag, never
    touches a file's status or its analysis. Captures that fall outside the
    window keep everything they already have -- excluding an era is a statement
    about what is being studied, not a reason to destroy the record of it.

    Returns the resulting counts so a caller can report what changed.
    """
    conditions = []
    if project.scope_start:
        conditions.append(ArchivedFile.timestamp >= project.scope_start)
    if project.scope_end:
        conditions.append(ArchivedFile.timestamp < project.scope_end)

    # A capture with no timestamp cannot be placed on either side of the
    # boundary; keep it in scope rather than silently dropping it.
    in_scope = and_(*conditions) if conditions else true()
    if conditions:
        in_scope = or_(ArchivedFile.timestamp.is_(None), in_scope)

    await db.execute(
        update(ArchivedFile)
        .where(ArchivedFile.project_id == project.id)
        .values(in_scope=in_scope)
    )

    # URL exclusions: out of scope, and not fetched if not fetched yet. A
    # capture whose pattern was removed returns to the download queue.
    patterns = exclusions.parse(project.exclude_patterns)
    rows = (await db.execute(
        select(ArchivedFile.id, ArchivedFile.original_url, ArchivedFile.status,
               ArchivedFile.local_path, ArchivedFile.error_message)
        .where(ArchivedFile.project_id == project.id)
    )).all()
    out_ids, skip, requeue = [], {}, []
    for fid, url, status, local_path, msg in rows:
        hit = exclusions.match(url or "", patterns)
        was_excluded = (status == FileStatus.SKIPPED and (msg or "").startswith(exclusions.EXCLUDED_PREFIX))
        if hit:
            out_ids.append(fid)
            if local_path is None and status in (FileStatus.PENDING, FileStatus.ERROR, FileStatus.SKIPPED) \
                    and (status != FileStatus.SKIPPED or was_excluded):
                skip.setdefault(exclusions.reason(hit), []).append(fid)
        elif was_excluded:
            requeue.append(fid)
    for i in range(0, len(out_ids), 500):
        await db.execute(update(ArchivedFile).where(ArchivedFile.id.in_(out_ids[i:i + 500]))
                         .values(in_scope=False))
    for msg, ids in skip.items():
        for i in range(0, len(ids), 500):
            await db.execute(update(ArchivedFile).where(ArchivedFile.id.in_(ids[i:i + 500]))
                             .values(status=FileStatus.SKIPPED, error_message=msg))
    for i in range(0, len(requeue), 500):
        await db.execute(update(ArchivedFile).where(ArchivedFile.id.in_(requeue[i:i + 500]))
                         .values(status=FileStatus.PENDING, error_message=None))
    await db.commit()

    rows = await db.execute(
        select(ArchivedFile.in_scope, func.count(ArchivedFile.id))
        .where(ArchivedFile.project_id == project.id)
        .group_by(ArchivedFile.in_scope)
    )
    counts = {bool(flag): int(n) for flag, n in rows.all()}
    return {"in_scope": counts.get(True, 0), "excluded": counts.get(False, 0),
            "by_pattern": len(out_ids), "requeued": len(requeue)}


async def sync_project_counts(db: AsyncSession, project: Project) -> Project:
    """Sync project file counts from actual file statuses in database.

    Only writes/commits when a count actually changed, to avoid a needless
    write on every read (costly when the DB lives on slow/NFS storage).
    """
    counts = await count_by_status(db, project.id)

    new_total = sum(counts.values())
    # A file whose analysis failed is still downloaded (see is_downloaded()).
    errors_with_file = 0
    if counts.get("ERROR"):
        errors_with_file = (await db.execute(
            select(func.count(ArchivedFile.id))
            .where(ArchivedFile.project_id == project.id)
            .where(ArchivedFile.status == FileStatus.ERROR)
            .where(ArchivedFile.local_path.isnot(None))
        )).scalar() or 0
    new_downloaded = (
        counts.get("DOWNLOADED", 0) + counts.get("PROCESSING", 0)
        + counts.get("PROCESSED", 0) + counts.get("SKIPPED", 0) + errors_with_file
    )
    new_processed = counts.get("PROCESSED", 0)

    if (project.files_total, project.files_downloaded, project.files_processed) != (
        new_total, new_downloaded, new_processed
    ):
        project.files_total = new_total
        project.files_downloaded = new_downloaded
        project.files_processed = new_processed
        await db.commit()

    return project


async def reconcile_project_status(db: AsyncSession, project: Project) -> Project:
    """Repair a project status that the data contradicts.

    Two cases matter. A project marked COMPLETED while files still await
    analysis is simply wrong (one project reached "completed" at 9 of 20,793
    analyzed). A project marked PROCESSING with no live run would make the
    workflow fragment poll forever, so this is also the backstop that
    guarantees the HTMX poll self-terminates if a worker dies abruptly.
    """
    from ..services.processing import processing_worker

    counts = await count_by_status(db, project.id)
    pending = counts.get("DOWNLOADED", 0) + counts.get("PROCESSING", 0)
    new_status = None

    if project.status == ProjectStatus.COMPLETED and pending > 0:
        new_status = ProjectStatus.DOWNLOAD_COMPLETE
    elif project.status == ProjectStatus.PROCESSING:
        progress = await processing_worker.get_progress(project.id)
        if not progress.get("running"):
            new_status = ProjectStatus.DOWNLOAD_COMPLETE
    elif project.status == ProjectStatus.DOWNLOADING:
        # A download task lives in this process only; after a restart or a
        # crash the status is all that is left of it.
        task = download_service.active_downloads.get(project.id)
        if task is None or task.done():
            new_status = (ProjectStatus.URLS_READY if counts.get("PENDING", 0)
                          else ProjectStatus.DOWNLOAD_COMPLETE)

    if new_status is not None and new_status != project.status:
        project.status = new_status
        await db.commit()

    return project


async def reset_failed_processing(db: AsyncSession, project_id: int) -> dict:
    """Reset LLM failures and stranded claims back to DOWNLOADED."""
    failed_count = (await db.execute(
        select(func.count(ArchivedFile.id)).where(
            ArchivedFile.project_id == project_id,
            ArchivedFile.status == FileStatus.ERROR,
            ArchivedFile.local_path.isnot(None),
        )
    )).scalar() or 0

    stuck_count = (await db.execute(
        select(func.count(ArchivedFile.id)).where(
            ArchivedFile.project_id == project_id,
            ArchivedFile.status == FileStatus.PROCESSING,
        )
    )).scalar() or 0

    if failed_count:
        await db.execute(
            update(ArchivedFile)
            .where(ArchivedFile.project_id == project_id)
            .where(ArchivedFile.status == FileStatus.ERROR)
            .where(ArchivedFile.local_path.isnot(None))
            .values(status=FileStatus.DOWNLOADED, error_message=None)
        )
    if stuck_count:
        await db.execute(
            update(ArchivedFile)
            .where(ArchivedFile.project_id == project_id)
            .where(ArchivedFile.status == FileStatus.PROCESSING)
            .values(status=FileStatus.DOWNLOADED)
        )
    if failed_count or stuck_count:
        await db.commit()

    total = failed_count + stuck_count
    return {
        "message": (f"Reset {total} files for reprocessing" if total
                    else "No failed processing to retry"),
        "reset_count": total,
        "failed_reset": failed_count,
        "stuck_reset": stuck_count,
    }


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(db: AsyncSession = Depends(get_db)):
    """List all projects"""
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()

    # Sync counts for all projects
    for project in projects:
        await sync_project_counts(db, project)

    return projects


@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate, db: AsyncSession = Depends(get_db)):
    """Create a new project"""
    db_project = Project(
        name=project.name,
        domain=project.domain,
        ollama_model=project.ollama_model,
        status=ProjectStatus.CREATED,
    )
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    return db_project


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get project details"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Sync counts from actual file statuses
    await sync_project_counts(db, project)
    return project


@router.delete("/{project_id}")
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a project and all its files"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await db.delete(project)
    await db.commit()
    return {"status": "deleted"}


@router.post("/{project_id}/search-wayback")
async def search_wayback(project_id: int, db: AsyncSession = Depends(get_db)):
    """Search Wayback Machine for available snapshots"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.status = ProjectStatus.FETCHING_URLS
    await db.commit()

    try:
        search_result = await wayback_service.search_domain(project.domain)

        project.total_snapshots = search_result["total_snapshots"]
        project.date_range_start = search_result["date_range_start"]
        project.date_range_end = search_result["date_range_end"]
        project.status = ProjectStatus.URLS_READY

        await db.commit()
        await db.refresh(project)

        return {
            "project": ProjectResponse.model_validate(project),
            "mimetypes": search_result["mimetypes"],
            "sample_urls": search_result["sample_urls"],
        }

    except Exception as e:
        project.status = ProjectStatus.ERROR
        await db.commit()
        # Several httpx errors stringify to "", which left the caller reading
        # "Failed to query Wayback Machine: " and no reason at all. Name the
        # exception class so a timeout is distinguishable from a refusal.
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}" if str(e)
                            else type(e).__name__)


@router.post("/{project_id}/configure-filters")
async def configure_filters(
    project_id: int,
    filters: ProjectFilterSettings,
    db: AsyncSession = Depends(get_db)
):
    """Configure download filters for a project"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.filter_date_start = filters.filter_date_start
    project.filter_date_end = filters.filter_date_end
    project.filter_frequency = filters.filter_frequency
    project.filter_mimetypes = filters.filter_mimetypes

    # Fetch filtered snapshots and create file records
    snapshots = await wayback_service.get_all_snapshots(
        project.domain,
        date_start=filters.filter_date_start,
        date_end=filters.filter_date_end,
        mimetypes=filters.filter_mimetypes,
    )

    # Apply frequency filter
    filtered_snapshots = wayback_service.filter_by_frequency(
        snapshots, filters.filter_frequency
    )

    # Clear existing files
    await db.execute(delete(ArchivedFile).where(ArchivedFile.project_id == project_id))

    # Create file records
    for snapshot in filtered_snapshots:
        file = ArchivedFile(
            project_id=project_id,
            original_url=snapshot["original_url"],
            timestamp=snapshot["timestamp"],
            mimetype=snapshot["mimetype"],
            status_code=snapshot["status_code"],
            status=FileStatus.PENDING,
        )
        db.add(file)

    project.files_total = len(filtered_snapshots)
    await db.commit()
    await db.refresh(project)

    return {
        "project": ProjectResponse.model_validate(project),
        "files_count": len(filtered_snapshots),
    }


@router.post("/{project_id}/start-download")
async def start_download(project_id: int, db: AsyncSession = Depends(get_db)):
    """Start downloading files for a project"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await download_service.start_download(project_id)
    return result


@router.post("/{project_id}/stop-download")
async def stop_download(project_id: int):
    """Stop downloading files for a project"""
    result = await download_service.stop_download(project_id)
    return result


@router.post("/{project_id}/retry-failed")
async def retry_failed_downloads(project_id: int, db: AsyncSession = Depends(get_db)):
    """Reset failed downloads to pending so they can be retried"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Count failed files
    count_result = await db.execute(
        select(func.count(ArchivedFile.id)).where(
            ArchivedFile.project_id == project_id,
            ArchivedFile.status == FileStatus.ERROR
        )
    )
    failed_count = count_result.scalar() or 0

    if failed_count == 0:
        return {"message": "No failed downloads to retry", "reset_count": 0}

    # Reset ERROR files back to PENDING
    await db.execute(
        update(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .where(ArchivedFile.status == FileStatus.ERROR)
        .values(status=FileStatus.PENDING, error_message=None)
    )
    await db.commit()

    return {"message": f"Reset {failed_count} failed downloads", "reset_count": failed_count}


@router.get("/{project_id}/download-progress")
async def get_download_progress(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get current download progress"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    progress = await download_service.get_progress(project_id)
    return {
        "project_status": project.status,
        **progress,
    }


@router.post("/{project_id}/start-processing")
async def start_processing(project_id: int, db: AsyncSession = Depends(get_db)):
    """Start LLM processing for downloaded files"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await llm_service.start_processing(project_id)
    return result


@router.post("/{project_id}/stop-processing")
async def stop_processing(project_id: int):
    """Stop LLM processing"""
    result = await llm_service.stop_processing(project_id)
    return result


@router.get("/{project_id}/processing-progress")
async def get_processing_progress(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get current processing progress"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    progress = await llm_service.get_progress(project_id)

    # Count files pending LLM processing (DOWNLOADED status - ready for processing)
    pending_result = await db.execute(
        select(func.count(ArchivedFile.id)).where(
            ArchivedFile.project_id == project_id,
            ArchivedFile.status == FileStatus.DOWNLOADED
        )
    )
    pending_count = pending_result.scalar() or 0

    # Count files stuck in processing (interrupted)
    processing_result = await db.execute(
        select(func.count(ArchivedFile.id)).where(
            ArchivedFile.project_id == project_id,
            ArchivedFile.status == FileStatus.PROCESSING
        )
    )
    processing_count = processing_result.scalar() or 0

    # Count files that failed LLM processing (have local_path but ERROR status)
    error_result = await db.execute(
        select(func.count(ArchivedFile.id)).where(
            ArchivedFile.project_id == project_id,
            ArchivedFile.status == FileStatus.ERROR,
            ArchivedFile.local_path.isnot(None)
        )
    )
    error_count = error_result.scalar() or 0

    return {
        "project_status": project.status,
        "pending": pending_count,
        "processing_stuck": processing_count,
        "errors": error_count,
        **progress,
    }


@router.post("/{project_id}/retry-failed-processing")
async def retry_failed_processing(project_id: int, db: AsyncSession = Depends(get_db)):
    """Reset failed LLM processing to downloaded so they can be retried"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return await reset_failed_processing(db, project_id)


@router.get("/{project_id}/files", response_model=List[FileResponse])
async def list_files(
    project_id: int,
    status: Optional[FileStatus] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """List files for a project"""
    query = select(ArchivedFile).where(ArchivedFile.project_id == project_id)

    if status:
        query = query.where(ArchivedFile.status == status)

    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    files = result.scalars().all()
    return files


@router.get("/{project_id}/results")
async def get_results(
    project_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get processed files with their analysis"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get files with analysis
    query = (
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .where(ArchivedFile.status == FileStatus.PROCESSED)
        .options(selectinload(ArchivedFile.analysis))
        .offset((page - 1) * per_page)
        .limit(per_page)
    )

    result = await db.execute(query)
    files = result.scalars().all()

    # Count total
    count_query = (
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .where(ArchivedFile.status == FileStatus.PROCESSED)
    )
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "results": [
            {
                "file": FileResponse.model_validate(f),
                "analysis": AnalysisResponse.model_validate(f.analysis) if f.analysis else None,
            }
            for f in files
        ],
    }


@router.post("/{project_id}/search")
async def search_results(
    project_id: int,
    query: SearchQuery,
    db: AsyncSession = Depends(get_db)
):
    """Search through processed results"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Build query
    base_query = (
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .where(ArchivedFile.status == FileStatus.PROCESSED)
        .options(selectinload(ArchivedFile.analysis))
    )

    result = await db.execute(base_query)
    files = result.scalars().all()

    # Filter in Python (for JSON fields)
    filtered = []
    for f in files:
        if not f.analysis:
            continue

        # Text search in description
        if query.query:
            if not f.analysis.description or query.query.lower() not in f.analysis.description.lower():
                continue

        # Filter by people
        if query.people:
            people = f.analysis.people_mentioned or []
            if not any(p.lower() in [x.lower() for x in people] for p in query.people):
                continue

        # Filter by subjects
        if query.subjects:
            subjects = f.analysis.subjects or []
            if not any(s.lower() in [x.lower() for x in subjects] for s in query.subjects):
                continue

        # Filter by sentiment
        if query.sentiment:
            if f.analysis.sentiment != query.sentiment:
                continue

        # Filter by content type
        if query.content_type:
            if f.analysis.content_type != query.content_type:
                continue

        # Filter by tags
        if query.tags:
            tags = f.analysis.tags or []
            if not any(t.lower() in [x.lower() for x in tags] for t in query.tags):
                continue

        filtered.append(f)

    # Paginate
    total = len(filtered)
    start = (query.page - 1) * query.per_page
    end = start + query.per_page
    paginated = filtered[start:end]

    return {
        "total": total,
        "page": query.page,
        "per_page": query.per_page,
        "pages": (total + query.per_page - 1) // query.per_page,
        "results": [
            {
                "file": FileResponse.model_validate(f),
                "analysis": AnalysisResponse.model_validate(f.analysis) if f.analysis else None,
            }
            for f in paginated
        ],
    }
