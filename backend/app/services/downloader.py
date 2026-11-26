"""Download service for fetching archived content from Wayback Machine"""
import os
import asyncio
import aiofiles
import httpx
from typing import Optional, Callable, Any
from datetime import datetime
from urllib.parse import urlparse, unquote
import re
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, func, or_, select, update

from ..models import Project, ArchivedFile, FileStatus, ProjectStatus

log = logging.getLogger("app.download")
from ..database import async_session_maker, DATABASE_DIR


# ERROR covers two different things: a download that failed (no file) and an
# analysis that failed on a file that downloaded fine. Only the first is
# missing from the corpus; counting both made the Download step show analysis
# failures as files still to fetch.
_HELD_STATUSES = (FileStatus.DOWNLOADED, FileStatus.PROCESSING,
                  FileStatus.PROCESSED, FileStatus.SKIPPED)


def is_downloaded():
    """SQL condition: the capture has been fetched (or needs no fetching)."""
    return or_(ArchivedFile.status.in_(_HELD_STATUSES),
               and_(ArchivedFile.status == FileStatus.ERROR,
                    ArchivedFile.local_path.isnot(None)))


def is_download_error():
    """SQL condition: the fetch itself failed."""
    return and_(ArchivedFile.status == FileStatus.ERROR, ArchivedFile.local_path.is_(None))


def archive_error_code(status_code: Optional[str]) -> Optional[int]:
    """The HTTP error the archive recorded for a capture, or None.

    The CDX index stores status codes as strings ("200", "404", "-" for
    revisits). A 4xx/5xx capture is the archive's copy of an error page, so
    fetching it yields nothing and calling it a failed download misreports a
    loss that happened at crawl time, not here.
    """
    try:
        code = int(status_code or "")
    except ValueError:
        return None
    return code if code >= 400 else None


def archive_error_message(code: int) -> str:
    return f"Archive only holds an error page (HTTP {code})"


def redirect_lost_message(capture_code: str, fetch_code: int) -> Optional[str]:
    """A 3xx capture whose target 404s was a redirect to a page the archive
    never captured. Retrying cannot help, so it is a skip, not a failure."""
    if (capture_code or "").startswith("3") and fetch_code in (403, 404):
        return f"Redirect target was never archived (capture HTTP {capture_code})"
    return None


class DownloadService:
    """Service for downloading archived files from Wayback Machine"""

    WAYBACK_BASE_URL = "http://web.archive.org/web"
    # Store downloads alongside the database (backend/data/), which is the directory
    # mounted as a persistent volume in docker-compose. Reusing DATABASE_DIR keeps the
    # data location in one place so downloads survive container rebuilds.
    PROJECTS_DIR = os.path.join(DATABASE_DIR, "projects")

    def __init__(self):
        self.active_downloads = {}  # project_id -> task
        self.download_progress = {}  # project_id -> progress info
        os.makedirs(self.PROJECTS_DIR, exist_ok=True)

    def get_project_dir(self, project_id: int, project_name: str) -> str:
        """Get the directory for a project's downloaded files"""
        safe_name = re.sub(r'[^\w\-]', '_', project_name)
        project_dir = os.path.join(self.PROJECTS_DIR, f"{project_id}_{safe_name}")
        os.makedirs(project_dir, exist_ok=True)
        return project_dir

    def get_local_path(self, project_dir: str, original_url: str, timestamp: str) -> str:
        """Generate local file path for a downloaded file"""
        parsed = urlparse(original_url)
        path = unquote(parsed.path) or "/index"
        if path.endswith("/"):
            path += "index.html"

        # Clean up path
        path = re.sub(r'[<>:"|?*]', '_', path)
        path = path.lstrip("/")

        # Add timestamp prefix to avoid collisions
        filename = f"{timestamp}_{path.replace('/', '_')}"
        if not filename.endswith(('.html', '.htm')):
            filename += '.html'

        return os.path.join(project_dir, filename)

    async def start_download(self, project_id: int, progress_callback: Optional[Callable] = None):
        """Start downloading files for a project"""
        if project_id in self.active_downloads:
            return {"error": "Download already in progress"}

        task = asyncio.create_task(self._download_project(project_id, progress_callback))
        self.active_downloads[project_id] = task
        return {"status": "started"}

    async def stop_download(self, project_id: int):
        """Stop an active download"""
        if project_id in self.active_downloads:
            self.active_downloads[project_id].cancel()
            del self.active_downloads[project_id]
            return {"status": "stopped"}
        return {"error": "No active download"}

    async def get_progress(self, project_id: int) -> dict:
        """Get current download progress from database"""
        # If we have active progress info, return it
        if project_id in self.download_progress:
            return self.download_progress[project_id]

        # Otherwise, calculate from database
        async with async_session_maker() as session:
            # Get total files count
            total_result = await session.execute(
                select(func.count(ArchivedFile.id)).where(
                    ArchivedFile.project_id == project_id
                )
            )
            total = total_result.scalar() or 0

            downloaded_result = await session.execute(
                select(func.count(ArchivedFile.id)).where(
                    ArchivedFile.project_id == project_id, is_downloaded()
                )
            )
            downloaded = downloaded_result.scalar() or 0

            # Get pending files count
            pending_result = await session.execute(
                select(func.count(ArchivedFile.id)).where(
                    ArchivedFile.project_id == project_id,
                    ArchivedFile.status == FileStatus.PENDING
                )
            )
            pending = pending_result.scalar() or 0

            # Download errors only; analysis failures have their file.
            error_result = await session.execute(
                select(func.count(ArchivedFile.id)).where(
                    ArchivedFile.project_id == project_id, is_download_error()
                )
            )
            errors = error_result.scalar() or 0

            return {
                "status": "idle",
                "downloaded": downloaded,
                "total": total,
                "pending": pending,
                "current_file": None,
                "errors": errors,
            }

    async def _download_project(self, project_id: int, progress_callback: Optional[Callable] = None):
        """Internal method to download all files for a project"""
        async with async_session_maker() as session:
            # Get project
            result = await session.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()

            if not project:
                return

            # Update project status
            project.status = ProjectStatus.DOWNLOADING
            await session.commit()

            # Get project directory
            project_dir = self.get_project_dir(project_id, project.name)

            # Get total files count for this project
            total_result = await session.execute(
                select(func.count(ArchivedFile.id)).where(
                    ArchivedFile.project_id == project_id
                )
            )
            total_files = total_result.scalar() or 0

            # Get already downloaded count
            already_downloaded_result = await session.execute(
                select(func.count(ArchivedFile.id)).where(
                    ArchivedFile.project_id == project_id, is_downloaded()
                )
            )
            already_downloaded = already_downloaded_result.scalar() or 0

            # Resume also retries fetches that failed last time (timeouts, 5xx,
            # throttling). Analysis failures keep their file and are not touched.
            await session.execute(
                update(ArchivedFile)
                .where(ArchivedFile.project_id == project_id, is_download_error())
                .values(status=FileStatus.PENDING, error_message=None)
                .execution_options(synchronize_session=False)
            )
            await session.commit()

            # Get files to download (only PENDING ones)
            result = await session.execute(
                select(ArchivedFile).where(
                    ArchivedFile.project_id == project_id,
                    ArchivedFile.status == FileStatus.PENDING
                )
            )
            files_to_download = result.scalars().all()

            downloaded = already_downloaded
            errors = 0
            session_downloaded = 0
            log.info("Download started: %d to fetch, %d of %d already held", len(files_to_download),
                     already_downloaded, total_files, extra={"project_id": project_id})

            self.download_progress[project_id] = {
                "status": "downloading",
                "downloaded": downloaded,
                "total": total_files,
                "pending": len(files_to_download),
                "current_file": None,
                "errors": 0,
            }

            # If no files to download, we're done
            if not files_to_download:
                project.status = ProjectStatus.DOWNLOAD_COMPLETE
                project.files_downloaded = downloaded
                await session.commit()
                self.download_progress[project_id]["status"] = "complete"
                if project_id in self.active_downloads:
                    del self.active_downloads[project_id]
                return

            # Rate limiting - Wayback Machine throttles aggressive requests
            rate_limit_delay = 1.0  # seconds between requests

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                for file in files_to_download:
                    try:
                        self.download_progress[project_id]["current_file"] = file.original_url

                        code = archive_error_code(file.status_code)
                        if code is not None:
                            file.status = FileStatus.SKIPPED
                            file.error_message = archive_error_message(code)
                            downloaded += 1
                            session_downloaded += 1
                            await session.commit()
                            continue   # nothing fetched, so no rate-limit pause

                        # Build Wayback URL
                        wayback_url = f"{self.WAYBACK_BASE_URL}/{file.timestamp}id_/{file.original_url}"
                        file.wayback_url = wayback_url

                        # Download content
                        response = await client.get(wayback_url)

                        if response.status_code == 200:
                            # Save file
                            local_path = self.get_local_path(project_dir, file.original_url, file.timestamp)
                            async with aiofiles.open(local_path, 'wb') as f:
                                await f.write(response.content)

                            file.local_path = local_path
                            file.file_size = len(response.content)
                            file.status = FileStatus.DOWNLOADED
                            file.downloaded_at = datetime.utcnow()
                            downloaded += 1
                            session_downloaded += 1
                        elif lost := redirect_lost_message(file.status_code, response.status_code):
                            file.status = FileStatus.SKIPPED
                            file.error_message = lost
                            downloaded += 1
                            session_downloaded += 1
                        else:
                            file.status = FileStatus.ERROR
                            file.error_message = f"HTTP {response.status_code}"
                            errors += 1
                            log.warning("Download failed (HTTP %s): %s @ %s", response.status_code,
                                        file.original_url, file.timestamp, extra={"project_id": project_id})

                    except asyncio.CancelledError:
                        log.info("Download stopped by request", extra={"project_id": project_id})
                        # Download was stopped
                        project.status = ProjectStatus.URLS_READY
                        project.files_downloaded = downloaded
                        await session.commit()
                        if project_id in self.active_downloads:
                            del self.active_downloads[project_id]
                        return

                    except Exception as e:
                        file.status = FileStatus.ERROR
                        file.error_message = str(e)
                        errors += 1
                        log.warning("Download failed (%s: %s): %s @ %s", type(e).__name__, e,
                                    file.original_url, file.timestamp, extra={"project_id": project_id})

                    await session.commit()

                    # Update progress
                    project.files_downloaded = downloaded
                    self.download_progress[project_id].update({
                        "downloaded": downloaded,
                        "pending": len(files_to_download) - session_downloaded - errors,
                        "errors": errors,
                    })

                    if progress_callback:
                        await progress_callback(self.download_progress[project_id])

                    # Rate limiting
                    await asyncio.sleep(rate_limit_delay)

            log.info("Download finished: %d fetched this run, %d failed", session_downloaded, errors,
                     extra={"project_id": project_id})
            # Make the new pages searchable. Reads local files only, no network.
            from .search_index import start_index
            start_index(project_id)
            # Update final status
            project.status = ProjectStatus.DOWNLOAD_COMPLETE
            project.files_downloaded = downloaded
            await session.commit()

            self.download_progress[project_id]["status"] = "complete"
            self.download_progress[project_id]["current_file"] = None

            if project_id in self.active_downloads:
                del self.active_downloads[project_id]


# Global instance
download_service = DownloadService()
