"""API routes for exporting data"""
import csv
import json
import io
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models import Project, ArchivedFile, ArchivalCard, FileAnalysis, FileStatus
from ..services import enrichment

router = APIRouter(prefix="/export", tags=["export"])


def card_to_dict(card):
    """Convert an archival card to a dictionary for export."""
    if not card:
        return None
    return {
        "registrar": card.registrar,
        "whois_created": card.whois_created,
        "whois_expires": card.whois_expires,
        "whois_updated": card.whois_updated,
        "nameservers": card.nameservers,
        "whois_status": card.whois_status,
        "registrant": card.registrant,
        "whois_source": card.whois_source,
        "whois_domain": card.whois_domain,
        "tech_stack": card.tech_stack,
        "timeline": card.timeline,
        "likely_people": card.likely_people,
        "llm_summary": card.llm_summary,
        "llm_model_used": card.llm_model_used,
        "generated_at": card.generated_at.isoformat() if card.generated_at else None,
    }


def file_to_dict(file, include_raw: bool = False):
    """Convert file and analysis to dictionary for export"""
    data = {
        "id": file.id,
        "original_url": file.original_url,
        "wayback_url": file.wayback_url,
        "timestamp": file.timestamp,
        "mimetype": file.mimetype,
        "file_size": file.file_size,
        "status": file.status.value if file.status else None,
        "downloaded_at": file.downloaded_at.isoformat() if file.downloaded_at else None,
        "processed_at": file.processed_at.isoformat() if file.processed_at else None,
    }

    if file.analysis:
        data["analysis"] = {
            "file_path": file.analysis.file_path,
            "estimated_date": file.analysis.estimated_date,
            "date_confidence": file.analysis.date_confidence,
            "people_mentioned": file.analysis.people_mentioned,
            "interaction_type": file.analysis.interaction_type,
            "subjects": file.analysis.subjects,
            "sentiment": file.analysis.sentiment,
            "sentiment_score": file.analysis.sentiment_score,
            "content_type": file.analysis.content_type,
            "tags": file.analysis.tags,
            "description": file.analysis.description,
            "model_used": file.analysis.model_used,
            "created_at": file.analysis.created_at.isoformat() if file.analysis.created_at else None,
        }
        if include_raw:
            data["analysis"]["raw_response"] = file.analysis.raw_response
    else:
        data["analysis"] = None

    return data


@router.get("/{project_id}/json")
async def export_json(
    project_id: int,
    include_raw: bool = Query(False, description="Include raw LLM responses"),
    processed_only: bool = Query(True, description="Only export processed files"),
    include_out_of_scope: bool = Query(False, description="Include captures outside the project's scope window"),
    db: AsyncSession = Depends(get_db)
):
    """Export project data as JSON"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get files. raw_response is by far the largest column and is dropped from
    # the output unless include_raw is set, so don't read it off disk either.
    analysis_load = selectinload(ArchivedFile.analysis)
    if not include_raw:
        analysis_load = analysis_load.defer(FileAnalysis.raw_response)
    query = (
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .options(analysis_load)
    )

    # An export describes the site under study. On a domain that changed hands
    # the excluded era is a different site entirely, so it is opt-in.
    if not include_out_of_scope:
        query = query.where(ArchivedFile.in_scope.is_(True))

    if processed_only:
        query = query.where(ArchivedFile.status == FileStatus.PROCESSED)

    result = await db.execute(query)
    files = result.scalars().all()

    # Fetch the archival card (if built)
    card_result = await db.execute(
        select(ArchivalCard).where(ArchivalCard.project_id == project_id)
    )
    card = card_result.scalar_one_or_none()

    # Build export data
    export_data = {
        "project": {
            "id": project.id,
            "name": project.name,
            "domain": project.domain,
            "status": project.status.value if project.status else None,
            "total_snapshots": project.total_snapshots,
            "date_range_start": project.date_range_start,
            "date_range_end": project.date_range_end,
            "files_total": project.files_total,
            "files_downloaded": project.files_downloaded,
            "files_processed": project.files_processed,
            "ollama_model": project.ollama_model,
            # Provenance: an export of a scoped project is a subset, and a
            # consumer needs to know which era it is holding.
            "scope_start": project.scope_start,
            "scope_end": project.scope_end,
            "scope_reason": project.scope_reason,
            "includes_out_of_scope": include_out_of_scope,
            "created_at": project.created_at.isoformat() if project.created_at else None,
        },
        "archival_card": card_to_dict(card),
        "files": [file_to_dict(f, include_raw) for f in files],
    }

    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

    # Return the body in one piece. StreamingResponse over an io.BytesIO
    # iterates it *by line*, and pretty-printed JSON is tens of thousands of
    # lines, so each one became a separate ASGI chunk -- that alone accounted
    # for ~7s of a ~7.5s export. The payload is already fully in memory here,
    # so there is nothing to stream.
    return Response(
        content=json_str.encode("utf-8"),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={project.name}_export.json"
        }
    )


@router.get("/{project_id}/csv")
async def export_csv(
    project_id: int,
    processed_only: bool = Query(True, description="Only export processed files"),
    include_out_of_scope: bool = Query(False, description="Include captures outside the project's scope window"),
    db: AsyncSession = Depends(get_db)
):
    """Export project data as CSV"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get files
    query = (
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        # raw_response is never emitted by this endpoint.
        .options(selectinload(ArchivedFile.analysis).defer(FileAnalysis.raw_response))
    )

    if not include_out_of_scope:
        query = query.where(ArchivedFile.in_scope.is_(True))

    if processed_only:
        query = query.where(ArchivedFile.status == FileStatus.PROCESSED)

    result = await db.execute(query)
    files = result.scalars().all()

    # Create CSV
    output = io.StringIO()
    fieldnames = [
        "id", "original_url", "wayback_url", "timestamp", "mimetype", "file_size",
        "status", "downloaded_at", "processed_at",
        "estimated_date", "date_confidence", "people_mentioned", "interaction_type",
        "subjects", "sentiment", "sentiment_score", "content_type", "tags", "description"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for file in files:
        row = {
            "id": file.id,
            "original_url": file.original_url,
            "wayback_url": file.wayback_url,
            "timestamp": file.timestamp,
            "mimetype": file.mimetype,
            "file_size": file.file_size,
            "status": file.status.value if file.status else None,
            "downloaded_at": file.downloaded_at.isoformat() if file.downloaded_at else None,
            "processed_at": file.processed_at.isoformat() if file.processed_at else None,
        }

        if file.analysis:
            row.update({
                "estimated_date": file.analysis.estimated_date,
                "date_confidence": file.analysis.date_confidence,
                "people_mentioned": json.dumps(file.analysis.people_mentioned) if file.analysis.people_mentioned else "",
                "interaction_type": file.analysis.interaction_type,
                "subjects": json.dumps(file.analysis.subjects) if file.analysis.subjects else "",
                "sentiment": file.analysis.sentiment,
                "sentiment_score": file.analysis.sentiment_score,
                "content_type": file.analysis.content_type,
                "tags": json.dumps(file.analysis.tags) if file.analysis.tags else "",
                "description": file.analysis.description,
            })

        writer.writerow(row)

    # Single body rather than a per-line stream, for the same reason as the
    # JSON export above.
    return Response(
        content=output.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={project.name}_export.csv"
        }
    )


@router.get("/{project_id}/stats")
async def get_stats(
    project_id: int,
    include_out_of_scope: bool = Query(False, description="Include captures outside the project's scope window"),
    db: AsyncSession = Depends(get_db),
):
    """Get statistics for a project"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all processed files with analysis
    query = (
        select(ArchivedFile)
        .where(ArchivedFile.project_id == project_id)
        .where(ArchivedFile.status == FileStatus.PROCESSED)
        # raw_response is never emitted by this endpoint.
        .options(selectinload(ArchivedFile.analysis).defer(FileAnalysis.raw_response))
    )

    if not include_out_of_scope:
        query = query.where(ArchivedFile.in_scope.is_(True))

    result = await db.execute(query)
    files = result.scalars().all()

    # Aggregate statistics
    sentiments = {}
    content_types = {}
    all_tags = {}
    all_people = set()
    all_subjects = set()
    from ..services.people import canonicalizer
    canon = await canonicalizer(db, project_id)

    for f in files:
        if not f.analysis:
            continue

        # Sentiment counts
        if f.analysis.sentiment:
            sentiments[f.analysis.sentiment] = sentiments.get(f.analysis.sentiment, 0) + 1

        # Content type counts
        if f.analysis.content_type:
            content_types[f.analysis.content_type] = content_types.get(f.analysis.content_type, 0) + 1

        # Tag counts
        for tag in (f.analysis.tags or []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

        # Unique people. Entries may be plain names or objects like
        # {"name": ...} depending on the model, so normalize before adding to a
        # set -- a raw dict is unhashable and used to raise a 500 here.
        for person in (f.analysis.people_mentioned or []):
            name = enrichment.person_to_str(person)
            if name:
                all_people.add(canon.name(name))

        # Unique subjects
        for subject in (f.analysis.subjects or []):
            if isinstance(subject, str) and subject:
                all_subjects.add(subject)

    return {
        "total_files": project.files_total,
        "downloaded": project.files_downloaded,
        "processed": project.files_processed,
        "sentiments": sentiments,
        "content_types": content_types,
        "top_tags": sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:20],
        "unique_people": sorted(list(all_people))[:50],
        "unique_subjects": sorted(list(all_subjects))[:50],
    }
