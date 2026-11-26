"""Concurrent, resumable LLM analysis worker.

Design notes worth knowing before changing this file:

* **One task owns the database session.** AsyncSession is not safe to share
  across concurrent tasks, and SQLite tolerates exactly one writer. Analyzer
  tasks therefore do HTTP and file I/O only; every read and write goes through
  the single ``_db_loop``. This removes a whole class of "database is locked"
  and identity-map bugs without introducing a broker.

* **Progress lives in the database** (``processing_runs``), not in a dict on a
  singleton. That is what lets the UI show a truthful number after a restart.

* **Dedupe is expressed in the claim query**, not planned up front, so it stays
  correct across restarts with no extra state to reconcile.

* **Duplicate captures get a copied analysis**, tagged with ``source_file_id``,
  rather than being marked SKIPPED. SKIPPED rows never count as processed, so
  the progress bar would cap partway and every capture but one would show as
  unanalyzed. Aggregate statistics must exclude copies -- see models.py.
"""
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_llm_config, resolve_local_path
from ..database import async_session_maker
from ..models import (
    ArchivedFile, FileAnalysis, FileStatus, Project, ProcessingRun, ProjectStatus,
)
from .llm import llm_service
from .llm_client import OllamaClient, RateLimiter

import logging as _logging
log = _logging.getLogger("app.analysis")

# A run whose heartbeat is older than this is treated as dead.
STALE_AFTER = timedelta(seconds=45)
HEARTBEAT_INTERVAL = 2.0

# Stop the run after this many consecutive failures: a dead endpoint should
# cost ~10 files, not the whole backlog.
CONSECUTIVE_FAILURE_LIMIT = 10

# How many LLM calls between quota checks when a usage ceiling is configured.
# The usage endpoint is cheap but not free to call, and quota moves slowly, so
# poll loosely when far from the ceiling and tighten as it approaches -- at ~0.08%
# per call a fixed 25-call gap can overshoot the target by ~2%.
USAGE_CHECK_EVERY = 25
USAGE_CHECK_NEAR = 5
USAGE_NEAR_MARGIN = 0.05


def _usage_check_interval(current: Optional[float], ceiling: float) -> int:
    """Calls to wait before re-checking quota, given the last reading."""
    if current is not None and ceiling - current <= USAGE_NEAR_MARGIN:
        return USAGE_CHECK_NEAR
    return USAGE_CHECK_EVERY

# Pre-flight: if more than this fraction of a sample is missing from disk,
# refuse to start rather than writing tens of thousands of ERROR rows.
PREFLIGHT_SAMPLE = 100
PREFLIGHT_MISSING_THRESHOLD = 0.5

MIN_CONTENT_CHARS = 50


@dataclass
class WorkItem:
    file_id: int
    url: str
    path: str
    size: Optional[int]
    timestamp: Optional[str]


@dataclass
class WorkResult:
    item: WorkItem
    analysis: Optional[Dict[str, Any]] = None
    raw: str = ""
    tokens: int = 0
    prompt_tokens: int = 0
    duration: float = 0.0
    error: Optional[str] = None
    fatal: bool = False
    skipped_reason: Optional[str] = None


_SENTINEL = object()


def _analyzable_sql_list() -> str:
    """SQL literal list of analyzable mimetypes (NULL handled separately)."""
    values = [m for m in llm_service.ANALYZABLE_MIMETYPES if m]
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


class ProcessingWorker:
    """Runs analysis for one project at a time, per project."""

    def __init__(self):
        self._tasks: Dict[int, asyncio.Task] = {}
        self._stop_events: Dict[int, asyncio.Event] = {}

    # ================= public API =================

    async def start(self, project_id: int, *, concurrency: Optional[int] = None,
                    model: Optional[str] = None) -> Dict[str, Any]:
        task = self._tasks.get(project_id)
        if task and not task.done():
            return {"status": "already_running"}

        cfg = get_llm_config()
        if concurrency:
            from dataclasses import replace
            cfg = replace(cfg, concurrency=max(1, min(64, concurrency)))

        # Fail fast on an unreachable endpoint instead of marking every file ERROR.
        health = await llm_service.check_ollama(cfg)
        if health.get("status") != "running":
            return {
                "status": "error",
                "message": f"Ollama endpoint not reachable at {cfg.safe_base_url()}: "
                           f"{health.get('message', 'unknown error')}",
            }

        async with async_session_maker() as session:
            project = await session.get(Project, project_id)
            if not project:
                return {"status": "error", "message": "Project not found"}
            chosen_model = model or project.ollama_model or cfg.model

            # The project's stored model may not exist on the current endpoint
            # (projects here were configured against a local Ollama holding
            # llama3.2:*, which ollama.com does not serve). Rather than failing
            # every file with a 404, fall back to the configured default.
            available = health.get("models") or []
            if available and chosen_model not in available:
                fallback = cfg.model if cfg.model in available else available[0]
                note = (f"Model '{chosen_model}' is not available on "
                        f"{cfg.safe_base_url()}; using '{fallback}' instead.")
                chosen_model = fallback
                project.ollama_model = fallback   # persist so the UI agrees
                await session.commit()
            else:
                note = None
                # A model picked in the start form becomes the project's model,
                # so close readings and link analysis (which read
                # project.ollama_model) use the same one.
                if model and project.ollama_model != chosen_model:
                    project.ollama_model = chosen_model
                    await session.commit()

            guard = await self._preflight(session, project_id)
            if guard is not None:
                return guard

        stop_event = asyncio.Event()
        self._stop_events[project_id] = stop_event
        self._tasks[project_id] = asyncio.create_task(
            self._run(project_id, cfg, chosen_model, stop_event)
        )
        result = {"status": "started", "model": chosen_model, "concurrency": cfg.concurrency}
        if note:
            result["note"] = note
        return result

    async def stop(self, project_id: int) -> Dict[str, Any]:
        event = self._stop_events.get(project_id)
        task = self._tasks.get(project_id)
        if not task or task.done():
            return {"status": "not_running"}
        if event:
            event.set()  # cooperative: stop claiming, let in-flight work drain
        return {"status": "stopping"}

    async def get_progress(self, project_id: int) -> Dict[str, Any]:
        async with async_session_maker() as session:
            return await self._progress(session, project_id)

    async def recover_orphans(self) -> None:
        """Clear state left behind by a process that died mid-run.

        Runs at startup, before traffic is served. Deliberately does not
        auto-resume: an unattended restart silently hammering a metered cloud
        endpoint is a bad surprise. The UI offers a Resume button instead.
        """
        now = datetime.utcnow()
        async with async_session_maker() as session:
            await session.execute(
                update(ProcessingRun)
                .where(ProcessingRun.status == "running")
                .values(status="interrupted", finished_at=now)
            )
            await session.execute(
                update(ArchivedFile)
                .where(ArchivedFile.status == FileStatus.PROCESSING)
                .values(status=FileStatus.DOWNLOADED)
            )
            await session.execute(
                update(Project)
                .where(Project.status == ProjectStatus.PROCESSING)
                .values(status=ProjectStatus.DOWNLOAD_COMPLETE)
            )
            await session.commit()

    # ================= progress =================

    async def _counts(self, session: AsyncSession, project_id: int) -> Dict[str, int]:
        """Status counts for the in-scope corpus only.

        These drive the analysis progress display, and analysis only ever runs
        on in-scope captures -- counting the excluded era here would show a
        denominator the worker will never work through.
        """
        rows = await session.execute(
            text("SELECT status, COUNT(*) FROM archived_files "
                 "WHERE project_id = :pid AND in_scope = 1 GROUP BY status"),
            {"pid": project_id},
        )
        return {str(status): int(count) for status, count in rows.fetchall()}

    async def _analyzable_remaining(self, session: AsyncSession, project_id: int) -> tuple:
        """Work still outstanding, as (distinct URLs, rows).

        Both numbers are needed and they are not interchangeable. Distinct URLs
        is the count of LLM calls left, so it drives the ETA. Rows is the count
        of files that will end up PROCESSED (each duplicate capture gets a
        copied analysis), so it is what the progress denominator must use --
        mixing the two makes the total drift upward as copies accumulate.
        """
        row = await session.execute(
            text(f"""
                SELECT COUNT(DISTINCT original_url), COUNT(*) FROM archived_files
                WHERE project_id = :pid AND status = 'DOWNLOADED'
                  AND in_scope = 1
                  AND local_path IS NOT NULL
                  AND (mimetype IN ({_analyzable_sql_list()}) OR mimetype IS NULL)
            """),
            {"pid": project_id},
        )
        urls, rows = row.fetchone()
        return int(urls or 0), int(rows or 0)

    async def _progress(self, session: AsyncSession, project_id: int) -> Dict[str, Any]:
        run = (await session.execute(
            select(ProcessingRun)
            .where(ProcessingRun.project_id == project_id)
            .order_by(ProcessingRun.id.desc())
            .limit(1)
        )).scalar_one_or_none()

        counts = await self._counts(session, project_id)
        remaining_urls, remaining_rows = await self._analyzable_remaining(session, project_id)
        processed = counts.get("PROCESSED", 0)
        errors_db = counts.get("ERROR", 0)
        skipped = counts.get("SKIPPED", 0)

        task = self._tasks.get(project_id)
        task_alive = bool(task and not task.done())

        if run is None:
            return {
                "status": "idle", "running": False, "stale": False,
                "processed": processed, "analyzed": 0, "copied": 0,
                "skipped": skipped, "errors": errors_db,
                "remaining_urls": remaining_urls,
                "analyzable_total": processed + remaining_rows,
                "current_file": None, "last_error": None,
                "model": None, "concurrency": None,
                "rate_per_min": None, "eta_seconds": None,
            }

        heartbeat = run.heartbeat_at or run.started_at or datetime.utcnow()
        heartbeat_lost = (datetime.utcnow() - heartbeat) > STALE_AFTER
        running = run.status == "running" and (task_alive or not heartbeat_lost)
        # "stale" drives the Resume affordance: either the process died and was
        # cleaned up at boot (status 'interrupted'), or a run claims to be
        # running but has stopped heartbeating and has no live task.
        stale = (
            (run.status in ("interrupted", "budget_reached"))
            or (run.status == "running" and heartbeat_lost and not task_alive)
        ) and remaining_urls > 0
        # A budget stop is a deliberate pause, not a failure: its message should
        # read as information rather than an error banner.
        paused_on_budget = run.status == "budget_reached"

        elapsed = ((run.finished_at or datetime.utcnow()) - run.started_at).total_seconds() \
            if run.started_at else 0.0
        analyzed = run.analyzed or 0
        rate = (analyzed / elapsed * 60.0) if elapsed > 1 and analyzed else None
        eta = int(remaining_urls / (rate / 60.0)) if rate and rate > 0 and remaining_urls else None

        return {
            "status": run.status,
            "running": running,
            "stale": stale and not task_alive,
            "processed": processed,
            "analyzed": analyzed,
            "copied": run.copied or 0,
            "skipped": skipped,
            "errors": errors_db,
            "run_errors": run.errors or 0,
            "remaining_urls": remaining_urls,
            "analyzable_total": processed + remaining_rows,
            "current_file": run.current_url if running else None,
            # Suppress a stale error banner once there is nothing left to do.
            "last_error": (run.last_error
                           if (running or remaining_urls) and not paused_on_budget else None),
            "paused_on_budget": paused_on_budget,
            "budget_note": run.last_error if paused_on_budget else None,
            "model": run.model_used,
            "concurrency": run.concurrency,
            "rate_per_min": round(rate, 1) if rate else None,
            "eta_seconds": eta,
        }

    # ================= pre-flight =================

    async def _preflight(self, session: AsyncSession, project_id: int) -> Optional[Dict[str, Any]]:
        """Refuse to start when the corpus is not actually on disk.

        Two projects in this database have ~43,000 rows marked DOWNLOADED whose
        files no longer exist. Without this check, one click would convert all
        of them into ERROR rows in a few minutes.
        """
        rows = (await session.execute(
            text("""
                SELECT local_path FROM archived_files
                WHERE project_id = :pid AND status = 'DOWNLOADED' AND in_scope = 1
                  AND local_path IS NOT NULL
                LIMIT :n
            """),
            {"pid": project_id, "n": PREFLIGHT_SAMPLE},
        )).fetchall()

        if not rows:
            return None

        paths = [r[0] for r in rows]

        def _count_missing() -> int:
            return sum(1 for p in paths if not os.path.exists(resolve_local_path(p)))

        missing = await asyncio.to_thread(_count_missing)
        ratio = missing / len(paths)
        if ratio > PREFLIGHT_MISSING_THRESHOLD:
            return {
                "status": "error",
                "message": (
                    f"{int(ratio * 100)}% of sampled files are missing from disk. "
                    "The download for this project needs to be re-run before analysis. "
                    "Refusing to start so the database is not filled with errors."
                ),
            }
        return None

    # ================= the run =================

    async def _run(self, project_id: int, cfg, model: str, stop_event: asyncio.Event) -> None:
        work_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.concurrency * 2)
        result_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.concurrency * 4)
        limiter = RateLimiter(cfg.requests_per_minute)
        run_id: Optional[int] = None
        analyzers: List[asyncio.Task] = []

        async with async_session_maker() as session:
            run = ProcessingRun(
                project_id=project_id, status="running", model_used=model,
                concurrency=cfg.concurrency, started_at=datetime.utcnow(),
                heartbeat_at=datetime.utcnow(),
            )
            session.add(run)
            await session.commit()
            run_id = run.id

            await session.execute(
                update(Project).where(Project.id == project_id)
                .values(status=ProjectStatus.PROCESSING)
            )
            # Reclaim rows stranded by an earlier interrupted run.
            await session.execute(
                update(ArchivedFile)
                .where(ArchivedFile.project_id == project_id)
                .where(ArchivedFile.status == FileStatus.PROCESSING)
                .values(status=FileStatus.DOWNLOADED)
            )
            await session.commit()

            await self._sweep_non_text(session, project_id)
            await self._copy_to_orphans(session, project_id)

        outcome = "completed"
        last_error: Optional[str] = None

        try:
            async with OllamaClient(cfg, limiter) as client:
                analyzers = [
                    asyncio.create_task(
                        self._analyzer(work_q, result_q, cfg, client, model)
                    )
                    for _ in range(cfg.concurrency)
                ]
                outcome, last_error = await self._db_loop(
                    project_id, run_id, cfg, model, work_q, result_q, stop_event,
                    analyzers, client,
                )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            outcome = "aborted"
            last_error = cfg.redact(f"{type(exc).__name__}: {exc}")
            log.exception("Analysis run crashed", extra={"project_id": project_id})
        finally:
            for t in analyzers:
                t.cancel()
            # shield: this must complete even when the task is being cancelled,
            # otherwise rows stay stranded in PROCESSING.
            await asyncio.shield(
                self._finalize(project_id, run_id, outcome, last_error)
            )
            self._tasks.pop(project_id, None)
            self._stop_events.pop(project_id, None)

    async def _sweep_non_text(self, session: AsyncSession, project_id: int) -> int:
        """Retire non-text files in one statement, without an LLM call each."""
        result = await session.execute(
            text(f"""
                UPDATE archived_files
                   SET status = 'SKIPPED',
                       error_message = 'Not a text document (' || COALESCE(mimetype, 'unknown') || ')'
                 WHERE project_id = :pid AND status = 'DOWNLOADED'
                   AND in_scope = 1
                   AND mimetype IS NOT NULL
                   AND mimetype NOT IN ({_analyzable_sql_list()})
            """),
            {"pid": project_id},
        )
        await session.commit()
        return result.rowcount or 0

    async def _claim(self, session: AsyncSession, project_id: int, limit: int) -> List[WorkItem]:
        """Claim a batch, one representative capture per URL.

        The representative is the newest capture of each URL. The CASE ordering
        keeps an already-PROCESSED or in-flight PROCESSING capture as the
        representative, which is what stops siblings from being claimed twice
        and preserves work done by earlier runs.

        Both halves filter in_scope. The inner one matters as much as the outer:
        if out-of-scope siblings could win the "newest capture" contest, a URL
        captured both before and after a domain takeover would be represented by
        the post-takeover page -- so the real page would never be analyzed and
        the spam version would be copied over it.
        """
        rows = (await session.execute(
            text(f"""
                SELECT a.id, a.original_url, a.local_path, a.file_size, a.timestamp
                FROM archived_files a
                WHERE a.project_id = :pid
                  AND a.status = 'DOWNLOADED'
                  AND a.in_scope = 1
                  AND a.local_path IS NOT NULL
                  AND (a.mimetype IN ({_analyzable_sql_list()}) OR a.mimetype IS NULL)
                  AND a.id = (
                      SELECT b.id FROM archived_files b
                      WHERE b.project_id = :pid
                        AND b.original_url = a.original_url
                        AND b.in_scope = 1
                        AND b.local_path IS NOT NULL
                        AND b.status IN ('DOWNLOADED', 'PROCESSING', 'PROCESSED')
                      ORDER BY CASE b.status
                                 WHEN 'PROCESSED'  THEN 0
                                 WHEN 'PROCESSING' THEN 1
                                 ELSE 2 END,
                               b.timestamp DESC, b.file_size DESC, b.id
                      LIMIT 1)
                ORDER BY a.id
                LIMIT :n
            """),
            {"pid": project_id, "n": limit},
        )).fetchall()

        if not rows:
            return []

        ids = [r[0] for r in rows]
        await session.execute(
            update(ArchivedFile)
            .where(ArchivedFile.id.in_(ids))
            .where(ArchivedFile.status == FileStatus.DOWNLOADED)
            .values(status=FileStatus.PROCESSING)
        )
        await session.commit()
        return [WorkItem(file_id=r[0], url=r[1], path=r[2], size=r[3], timestamp=r[4]) for r in rows]

    async def _analyzer(self, work_q: asyncio.Queue, result_q: asyncio.Queue,
                        cfg, client: OllamaClient, model: str) -> None:
        """Read, clean, and analyze. Never touches the database."""
        while True:
            item = await work_q.get()
            if item is _SENTINEL:
                work_q.task_done()
                break
            try:
                path = resolve_local_path(item.path)
                if not os.path.exists(path):
                    await result_q.put(WorkResult(item=item, error="Local file not found"))
                    continue
                if item.size and item.size > llm_service.MAX_FILE_SIZE:
                    await result_q.put(WorkResult(item=item, skipped_reason="File too large"))
                    continue

                text_content = await asyncio.to_thread(
                    llm_service.read_and_clean, item.path, cfg.max_content_chars
                )
                if len(text_content) < MIN_CONTENT_CHARS:
                    await result_q.put(WorkResult(item=item, skipped_reason="Too little content"))
                    continue

                res = await llm_service.analyze_content(
                    text_content, os.path.basename(path), item.url, cfg, client, model=model
                )
                # One retry at a lower temperature for an unparseable response
                # before giving up on the file.
                # A reply cut off at the output limit would only be cut off again.
                if (res.error and not res.fatal and res.analysis is None and res.raw_response
                        and res.done_reason != "length"):
                    res = await llm_service.analyze_content(
                        text_content, os.path.basename(path), item.url, cfg, client,
                        model=model, temperature=0.1,
                    )

                await result_q.put(WorkResult(
                    item=item, analysis=res.analysis, raw=res.raw_response,
                    tokens=res.tokens, prompt_tokens=res.prompt_tokens, duration=res.duration,
                    error=res.error, fatal=res.fatal,
                ))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await result_q.put(WorkResult(
                    item=item, error=cfg.redact(f"{type(exc).__name__}: {exc}")
                ))
            finally:
                work_q.task_done()

    async def _db_loop(self, project_id: int, run_id: int, cfg, model: str,
                       work_q: asyncio.Queue, result_q: asyncio.Queue,
                       stop_event: asyncio.Event, analyzers: List[asyncio.Task],
                       client: OllamaClient):
        """Sole owner of the database session for the duration of the run."""
        analyzed = copied = errors = skipped = 0
        consecutive_failures = 0
        outcome = "completed"
        last_error: Optional[str] = None
        # LLM calls actually issued this run (successes + failures), which is what
        # a metered endpoint bills against.
        calls_made = 0
        budget_note: Optional[str] = None
        usage_check_due = USAGE_CHECK_EVERY
        in_flight = 0
        batch_size = max(4, cfg.concurrency * 4)
        last_heartbeat = 0.0
        loop = asyncio.get_running_loop()

        async with async_session_maker() as session:
            while True:
                # Top up the queue unless we are stopping.
                want = batch_size - in_flight
                if cfg.max_calls_per_run:
                    # Never claim more than the budget can pay for, or the run
                    # overshoots the cap by a whole batch.
                    want = min(want, cfg.max_calls_per_run - calls_made - in_flight)
                if not stop_event.is_set() and want > 0:
                    items = await self._claim(session, project_id, want)
                    for it in items:
                        await work_q.put(it)
                        in_flight += 1
                    if not items and in_flight == 0:
                        break

                if in_flight == 0:
                    # Only a plain stop counts as a cancel; a budget stop (or an
                    # abort) has already recorded a more specific outcome.
                    if stop_event.is_set() and outcome == "completed":
                        outcome = "cancelled"
                    break

                try:
                    result: WorkResult = await asyncio.wait_for(result_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    await self._heartbeat(session, run_id, analyzed, copied, skipped,
                                          errors, None, force=True)
                    continue

                in_flight -= 1
                # A skip never reaches the endpoint, so it costs no quota.
                if not result.skipped_reason:
                    calls_made += 1
                    usage_check_due -= 1
                try:
                    if result.analysis is not None:
                        n_copies = await self._apply_success(session, project_id, result, model)
                        analyzed += 1
                        copied += n_copies
                        consecutive_failures = 0
                    elif result.skipped_reason:
                        await self._apply_skip(session, result)
                        skipped += 1
                        consecutive_failures = 0
                    else:
                        await self._apply_error(session, result)
                        errors += 1
                        last_error = result.error
                        log.warning("Analysis failed: %s: %s", result.item.url, result.error,
                                    extra={"project_id": project_id})
                        consecutive_failures += 1
                        if result.fatal:
                            outcome = "aborted"
                            stop_event.set()
                        elif consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                            outcome = "aborted"
                            last_error = (
                                f"Stopped after {consecutive_failures} consecutive failures. "
                                f"Last error: {result.error}"
                            )
                            stop_event.set()
                finally:
                    result_q.task_done()

                # ----- budget enforcement -----
                if cfg.max_calls_per_run and calls_made >= cfg.max_calls_per_run and not stop_event.is_set():
                    budget_note = (
                        f"Reached the configured limit of {cfg.max_calls_per_run} "
                        f"LLM calls for this run. Progress is saved — press Resume "
                        f"to continue when your quota allows."
                    )
                    outcome = "budget_reached"
                    stop_event.set()
                elif cfg.stop_at_session_usage and usage_check_due <= 0 and not stop_event.is_set():
                    usage = await client.fetch_usage()
                    # "max" is the tightest allowance the endpoint reports
                    # (session, weekly or monthly -- see fetch_usage).
                    current = (usage or {}).get("max")
                    usage_check_due = _usage_check_interval(current, cfg.stop_at_session_usage)
                    if current is not None and current >= cfg.stop_at_session_usage:
                        budget_note = (
                            f"Stopping at {current * 100:.0f}% of the endpoint's quota "
                            f"(limit set to {cfg.stop_at_session_usage * 100:.0f}%). "
                            f"Progress is saved — press Resume once the quota resets."
                        )
                        outcome = "budget_reached"
                        stop_event.set()

                now = loop.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    await self._heartbeat(session, run_id, analyzed, copied, skipped,
                                          errors, result.item.url)

            # Drain the analyzers.
            for _ in analyzers:
                await work_q.put(_SENTINEL)

            await self._heartbeat(session, run_id, analyzed, copied, skipped,
                                  errors, None, force=True)

        # A budget stop sets stop_event too, so do not relabel it as a cancel.
        if stop_event.is_set() and outcome == "completed":
            outcome = "cancelled"
        if budget_note:
            last_error = budget_note
        return outcome, last_error

    # ----- result application -----

    async def _apply_success(self, session: AsyncSession, project_id: int,
                             result: WorkResult, model: str) -> int:
        """Persist an analysis and copy it to the URL's other captures."""
        item = result.item
        now = datetime.utcnow()
        data = result.analysis

        await self._upsert_analysis(
            session, file_id=item.file_id, file_path=item.path, data=data,
            duration=result.duration, tokens=result.tokens,
            prompt_tokens=result.prompt_tokens,
            model=model, raw=result.raw, source_file_id=None,
        )
        await session.execute(
            update(ArchivedFile).where(ArchivedFile.id == item.file_id)
            .values(status=FileStatus.PROCESSED, processed_at=now, error_message=None,
                        last_raw_response=None)
        )

        # in_scope, so an excluded-era capture of the same URL does not receive a
        # copy of an analysis describing different content entirely.
        siblings = (await session.execute(
            text("""
                SELECT id, local_path FROM archived_files
                WHERE project_id = :pid AND original_url = :url
                  AND id != :rid AND status = 'DOWNLOADED' AND in_scope = 1
            """),
            {"pid": project_id, "url": item.url, "rid": item.file_id},
        )).fetchall()

        label = f"copied from capture {item.timestamp or item.file_id}"
        for sib_id, sib_path in siblings:
            await self._upsert_analysis(
                session, file_id=sib_id, file_path=sib_path, data=data,
                duration=0.0, tokens=0, model=f"{model} ({label})",
                raw=None, source_file_id=item.file_id,
            )
            await session.execute(
                update(ArchivedFile).where(ArchivedFile.id == sib_id)
                .values(status=FileStatus.PROCESSED, processed_at=now, error_message=None,
                        last_raw_response=None)
            )

        await session.commit()
        return len(siblings)

    async def _copy_to_orphans(self, session: AsyncSession, project_id: int) -> int:
        """Give waiting captures the analysis their URL already has.

        Copies to duplicate captures happen only when the representative is
        analyzed. A capture requeued later (by "Retry failed") whose URL was
        already analyzed is never claimed -- the claim query picks the
        PROCESSED representative -- so without this it waits forever.
        """
        rows = (await session.execute(text("""
            SELECT a.id AS orphan_id, a.local_path AS orphan_path,
                   MIN(src.id) AS src_id
            FROM archived_files a
            JOIN archived_files src
              ON src.project_id = a.project_id AND src.original_url = a.original_url
             AND src.status = 'PROCESSED' AND src.in_scope = 1
            JOIN file_analyses fa ON fa.file_id = src.id AND fa.source_file_id IS NULL
            WHERE a.project_id = :pid AND a.status = 'DOWNLOADED' AND a.in_scope = 1
            GROUP BY a.id
        """), {"pid": project_id})).mappings().all()
        now = datetime.utcnow()
        for r in rows:
            fa = (await session.execute(
                select(FileAnalysis).where(FileAnalysis.file_id == r["src_id"])
            )).scalar_one()
            src_ts = (await session.execute(
                select(ArchivedFile.timestamp).where(ArchivedFile.id == r["src_id"])
            )).scalar()
            data = {c: getattr(fa, c) for c in (
                "estimated_date", "date_confidence", "people_mentioned", "interaction_type",
                "subjects", "sentiment", "sentiment_score", "content_type", "tags", "description")}
            base_model = (fa.model_used or "").split(" (copied")[0]
            await self._upsert_analysis(
                session, file_id=r["orphan_id"], file_path=r["orphan_path"], data=data,
                duration=0.0, tokens=0,
                model=f"{base_model} (copied from capture {src_ts or r['src_id']})",
                raw=None, source_file_id=r["src_id"],
            )
            await session.execute(
                update(ArchivedFile).where(ArchivedFile.id == r["orphan_id"])
                .values(status=FileStatus.PROCESSED, processed_at=now, error_message=None,
                        last_raw_response=None)
            )
        if rows:
            await session.commit()
        return len(rows)

    async def _upsert_analysis(self, session: AsyncSession, *, file_id: int,
                               file_path: Optional[str], data: Dict[str, Any],
                               duration: float, tokens: int, model: str,
                               raw: Optional[str], source_file_id: Optional[int],
                               prompt_tokens: int = 0) -> None:
        """Insert or update. file_id is UNIQUE, so a plain insert would raise
        IntegrityError on any reprocess."""
        values = {
            "file_id": file_id,
            "file_path": file_path,
            "source_file_id": source_file_id,
            "estimated_date": data.get("estimated_date"),
            "date_confidence": data.get("date_confidence"),
            "people_mentioned": data.get("people_mentioned", []),
            "interaction_type": data.get("interaction_type"),
            "subjects": data.get("subjects", []),
            "sentiment": data.get("sentiment"),
            "sentiment_score": data.get("sentiment_score"),
            "content_type": data.get("content_type"),
            "tags": data.get("tags", []),
            "description": data.get("description"),
            "processing_time_seconds": duration,
            "tokens_used": tokens,
            "prompt_tokens": prompt_tokens,
            "model_used": model,
            "raw_response": raw,
            "created_at": datetime.utcnow(),
        }
        stmt = sqlite_insert(FileAnalysis).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values if k != "file_id"}
        await session.execute(
            stmt.on_conflict_do_update(index_elements=[FileAnalysis.file_id], set_=update_cols)
        )

    async def _apply_skip(self, session: AsyncSession, result: WorkResult) -> None:
        await session.execute(
            update(ArchivedFile).where(ArchivedFile.id == result.item.file_id)
            .values(status=FileStatus.SKIPPED, error_message=result.skipped_reason)
        )
        await session.commit()

    async def _apply_error(self, session: AsyncSession, result: WorkResult) -> None:
        await session.execute(
            update(ArchivedFile).where(ArchivedFile.id == result.item.file_id)
            .values(status=FileStatus.ERROR, error_message=(result.error or "")[:500],
                    last_raw_response=(result.raw or None) and result.raw[:4000])
        )
        await session.commit()

    # ----- bookkeeping -----

    async def _heartbeat(self, session: AsyncSession, run_id: int, analyzed: int,
                         copied: int, skipped: int, errors: int,
                         current_url: Optional[str], force: bool = False) -> None:
        await session.execute(
            update(ProcessingRun).where(ProcessingRun.id == run_id).values(
                analyzed=analyzed, copied=copied, skipped=skipped, errors=errors,
                current_url=current_url, heartbeat_at=datetime.utcnow(),
            )
        )
        await session.commit()

    async def _finalize(self, project_id: int, run_id: Optional[int],
                        outcome: str, last_error: Optional[str]) -> None:
        """Release claims and set a truthful terminal status."""
        (log.info if outcome in ("completed", "cancelled") else log.warning)(
            "Analysis run ended: %s%s", outcome, f" ({last_error})" if last_error else "",
            extra={"project_id": project_id})
        async with async_session_maker() as session:
            await session.execute(
                update(ArchivedFile)
                .where(ArchivedFile.project_id == project_id)
                .where(ArchivedFile.status == FileStatus.PROCESSING)
                .values(status=FileStatus.DOWNLOADED)
            )

            counts = await self._counts(session, project_id)
            remaining = counts.get("DOWNLOADED", 0) + counts.get("PROCESSING", 0)
            processed = counts.get("PROCESSED", 0)

            # Computed, never assumed. The old code set COMPLETED whenever the
            # loop ended, which is how a project reached "completed" at 9/20793.
            if outcome in ("cancelled", "aborted", "budget_reached"):
                status = ProjectStatus.DOWNLOAD_COMPLETE
            elif remaining == 0 and processed > 0:
                status = ProjectStatus.COMPLETED
            else:
                status = ProjectStatus.DOWNLOAD_COMPLETE

            await session.execute(
                update(Project).where(Project.id == project_id).values(status=status)
            )
            if run_id is not None:
                await session.execute(
                    update(ProcessingRun).where(ProcessingRun.id == run_id).values(
                        status=outcome, finished_at=datetime.utcnow(),
                        heartbeat_at=datetime.utcnow(),
                        last_error=(last_error or None),
                        current_url=None,
                    )
                )
            await session.commit()


processing_worker = ProcessingWorker()
