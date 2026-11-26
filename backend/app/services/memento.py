"""Ask other web archives for what the Wayback Machine never captured.

The Internet Archive is the largest web archive, not the only one. A linked
page or image it never captured may sit in a national or institutional
archive. Those speak Memento (RFC 7089): each publishes a TimeMap listing its
captures of a URL.

The Memento TimeTravel aggregator (timetravel.mementoweb.org) used to answer
for all of them at once; it is gone (NXDOMAIN since 2026). So this asks the
archives directly, like MemGator does. Only archives that answer automated
clients are listed: in November 2026 the Library of Congress, the UK Web
Archive, the Icelandic and Australian archives, Stanford and Perma refused
or put up a bot challenge.

Raw bytes are fetched with the ``id_`` modifier that these pywb/OpenWayback
archives share. A capture of a redirect is followed only while the redirect
stays inside the archive: following it out would fetch the live web, which
on a dead domain is usually a parking page.

Only targets the Wayback Machine has already answered for (``not_archived``)
are asked, once each (``memento_checked_at``). What is found is recorded with
``source_archive`` naming the archive, and ``wayback_url`` holding the
memento's address there.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

import aiofiles
import httpx
from sqlalchemy import text

from ..database import async_session_maker
from .links import (ANALYZABLE_PREFIXES, LINKED_DIR, JobState, RateLimiter, _file_extension,
                    _parse_ts)

log = logging.getLogger("app.memento")

ARCHIVES: List[Dict[str, str]] = [
    {"name": "Archive-It", "host": "wayback.archive-it.org",
     "timemap": "https://wayback.archive-it.org/all/timemap/link/{url}",
     "raw": "https://wayback.archive-it.org/all/{ts}id_/{url}",
     "view": "https://wayback.archive-it.org/all/{ts}/{url}"},
    {"name": "Arquivo.pt", "host": "arquivo.pt",
     "timemap": "https://arquivo.pt/wayback/timemap/link/{url}",
     "raw": "https://arquivo.pt/wayback/{ts}id_/{url}",
     "view": "https://arquivo.pt/wayback/{ts}/{url}"},
]
WAYBACK = "Wayback Machine"
CANDIDATES_PER_ARCHIVE = 3
REQUESTS_PER_MINUTE = 60
MAX_BYTES = 15_000_000
HEADERS = {"User-Agent": "DigitalArcheology/1.0 (historical research; Memento client)"}

_LINK = re.compile(r'<([^>]+)>\s*;([^<]*)')
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_TS_IN_URI = re.compile(r"/(\d{14})(?:[a-z]{2}_)?/(.+)$")


class ArchiveBusy(Exception):
    """An archive is throttling; back off and try the item again later."""


def parse_timemap(body: str) -> List[Dict[str, str]]:
    """Mementos from an application/link-format TimeMap: [{ts, uri, original}]."""
    out = []
    for uri, attrs in _LINK.findall(body or ""):
        fields = dict(_ATTR.findall(attrs))
        if "memento" not in fields.get("rel", "").split():
            continue
        ts = None
        m = _TS_IN_URI.search(uri)
        if m:
            ts = m.group(1)
        elif fields.get("datetime"):
            try:
                ts = parsedate_to_datetime(fields["datetime"]).strftime("%Y%m%d%H%M%S")
            except (TypeError, ValueError):
                ts = None
        if not ts:
            continue
        out.append({"ts": ts, "uri": uri, "original": m.group(2) if m else ""})
    return out


def by_distance(mementos: List[Dict[str, str]], ts: Optional[str]) -> List[Dict[str, str]]:
    target = _parse_ts(ts or "")
    if not target:
        return sorted(mementos, key=lambda m: m["ts"])

    def dist(m):
        t = _parse_ts(m["ts"])
        return abs((t - target).total_seconds()) if t else float("inf")
    return sorted(mementos, key=dist)


async def _get_inside(client: httpx.AsyncClient, url: str, host: str) -> Optional[httpx.Response]:
    """GET, following redirects only while they stay on the archive's host."""
    for _ in range(6):
        response = await client.get(url)
        if response.status_code in (429, 503):
            raise ArchiveBusy(f"{host} HTTP {response.status_code}")
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        location = response.headers.get("location")
        if not location:
            return None
        url = urljoin(url, location)
        if urlsplit(url).hostname != host:
            return None          # the capture redirects to the live web
    return None


async def find(client: httpx.AsyncClient, pace: RateLimiter, url: str, ts: Optional[str],
               want_image: bool = False) -> Optional[Dict[str, Any]]:
    """The closest capture of ``url`` held by another archive, with its bytes."""
    for archive in ARCHIVES:
        await pace.acquire()
        try:
            response = await client.get(archive["timemap"].format(url=url))
        except (httpx.TimeoutException, httpx.TransportError):
            continue
        if response.status_code in (429, 503):
            raise ArchiveBusy(f"{archive['name']} HTTP {response.status_code}")
        if response.status_code != 200:
            continue
        for memento in by_distance(parse_timemap(response.text), ts)[:CANDIDATES_PER_ARCHIVE]:
            original = memento["original"] or url
            await pace.acquire()
            try:
                raw = await _get_inside(client, archive["raw"].format(ts=memento["ts"], url=original),
                                        archive["host"])
            except (httpx.TimeoutException, httpx.TransportError):
                continue
            if raw is None or raw.status_code != 200 or not raw.content:
                continue
            mimetype = (raw.headers.get("content-type") or "").split(";")[0].strip().lower()
            if want_image and not mimetype.startswith("image/"):
                continue
            if len(raw.content) > MAX_BYTES:
                continue
            final = _TS_IN_URI.search(str(raw.url))
            got_ts = final.group(1) if final else memento["ts"]
            return {"archive": archive["name"], "timestamp": got_ts, "content": raw.content,
                    "mimetype": mimetype, "original": original,
                    "view_url": archive["view"].format(ts=got_ts, url=original)}
    return None


# ---------- jobs ----------

_jobs: Dict[tuple, JobState] = {}


def job(project_id: int, kind: str = "links-memento") -> JobState:
    return _jobs.get((project_id, kind)) or JobState(kind=kind)


async def pending(db, project_id: int, kind: str = "links") -> int:
    table = "link_targets" if kind == "links" else "page_images"
    return (await db.execute(text(
        f"SELECT COUNT(*) FROM {table} WHERE project_id = :pid AND recovery_status = 'not_archived' "
        "AND memento_checked_at IS NULL"), {"pid": project_id})).scalar() or 0


async def recover_links(project_id: int, state: Optional[JobState] = None, limit: int = 0) -> Dict[str, Any]:
    state = state or JobState(kind="links-memento")
    out_dir = os.path.join(LINKED_DIR, str(project_id))
    os.makedirs(out_dir, exist_ok=True)
    async with async_session_maker() as db:
        rows = (await db.execute(text("""
            SELECT id, url, first_cited_ts, analysis_status FROM link_targets
            WHERE project_id = :pid AND recovery_status = 'not_archived' AND memento_checked_at IS NULL
            ORDER BY CASE kind WHEN 'reference' THEN 0 WHEN 'article' THEN 1 WHEN 'document' THEN 2 ELSE 3 END,
                     cited_by_pages DESC, id
        """), {"pid": project_id})).fetchall()
    if limit:
        rows = rows[:limit]
    state.total = len(rows)
    found = 0
    pace = RateLimiter(REQUESTS_PER_MINUTE)
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0), headers=HEADERS,
                                 follow_redirects=False) as client:
        for target_id, url, cited_ts, analysis_status in rows:
            if state.stop.is_set():
                break
            state.current = url
            hit = await _find_with_backoff(client, pace, url, cited_ts, state)
            if hit is None and state.stop.is_set():
                break
            values: Dict[str, Any] = {"memento_checked_at": datetime.utcnow()}
            if hit:
                path = os.path.join(out_dir, f"{target_id}{_file_extension(hit['mimetype'], hit['original'])}")
                async with aiofiles.open(path, "wb") as fh:
                    await fh.write(hit["content"])
                cited, captured = _parse_ts(cited_ts or ""), _parse_ts(hit["timestamp"])
                values.update(
                    recovery_status="recovered", source_archive=hit["archive"], error_message=None,
                    wayback_timestamp=hit["timestamp"], wayback_url=hit["view_url"],
                    capture_distance_days=abs((captured - cited).days) if cited and captured else None,
                    mimetype=hit["mimetype"] or None, local_path=path, file_size=len(hit["content"]),
                    recovered_at=datetime.utcnow())
                if hit["mimetype"].startswith(ANALYZABLE_PREFIXES) or not hit["mimetype"]:
                    if analysis_status in (None, "not_selected"):
                        values["analysis_status"] = "pending"
                else:
                    values["analysis_status"] = "skipped"
                found += 1
                log.info("Recovered from %s: %s", hit["archive"], url, extra={"project_id": project_id})
            sets = ", ".join(f"{k} = :{k}" for k in values)
            async with async_session_maker() as db:
                await db.execute(text(f"UPDATE link_targets SET {sets} WHERE id = :id"), {**values, "id": target_id})
                await db.commit()
            state.done += 1
            state.message = f"{found} found in other archives"
    return {"checked": state.done, "found": found}


async def recover_images(project_id: int, state: Optional[JobState] = None, limit: int = 0) -> Dict[str, Any]:
    from .images import DESCRIBABLE, IMAGES_DIR, MIN_IMAGE_BYTES
    state = state or JobState(kind="img-memento")
    out_dir = os.path.join(IMAGES_DIR, str(project_id))
    os.makedirs(out_dir, exist_ok=True)
    async with async_session_maker() as db:
        rows = (await db.execute(text("""
            SELECT pi.src_url, MIN(af.timestamp) FROM page_images pi
            JOIN archived_files af ON af.id = pi.file_id
            WHERE pi.project_id = :pid AND pi.recovery_status = 'not_archived' AND pi.memento_checked_at IS NULL
            GROUP BY pi.src_url ORDER BY COUNT(*) DESC, pi.src_url
        """), {"pid": project_id})).fetchall()
    if limit:
        rows = rows[:limit]
    state.total = len(rows)
    found = 0
    pace = RateLimiter(REQUESTS_PER_MINUTE)
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0), headers=HEADERS,
                                 follow_redirects=False) as client:
        for index, (src, page_ts) in enumerate(rows):
            if state.stop.is_set():
                break
            state.current = src
            hit = await _find_with_backoff(client, pace, src, page_ts, state, want_image=True)
            if hit is None and state.stop.is_set():
                break
            values: Dict[str, Any] = {"memento_checked_at": datetime.utcnow()}
            if hit and len(hit["content"]) >= MIN_IMAGE_BYTES:
                path = os.path.join(out_dir, f"m{index:05d}{_file_extension(hit['mimetype'], src)}")
                async with aiofiles.open(path, "wb") as fh:
                    await fh.write(hit["content"])
                values.update(recovery_status="recovered", source_archive=hit["archive"], error_message=None,
                              wayback_timestamp=hit["timestamp"], wayback_url=hit["view_url"],
                              mimetype=hit["mimetype"], local_path=path, file_size=len(hit["content"]),
                              description_status="pending" if hit["mimetype"] in DESCRIBABLE else "skipped")
                found += 1
            sets = ", ".join(f"{k} = :{k}" for k in values)
            async with async_session_maker() as db:
                await db.execute(text(f"UPDATE page_images SET {sets} WHERE project_id = :pid AND src_url = :src"),
                                 {**values, "pid": project_id, "src": src})
                await db.commit()
            state.done += 1
            state.message = f"{found} found in other archives"
    return {"checked": state.done, "found": found}


async def _find_with_backoff(client, pace, url, ts, state: JobState, want_image: bool = False):
    delay = 30.0
    for _ in range(6):
        try:
            return await find(client, pace, url, ts, want_image=want_image)
        except ArchiveBusy as exc:
            state.message = f"{exc}; waiting {int(delay)}s"
            try:
                await asyncio.wait_for(state.stop.wait(), timeout=delay)
                return None
            except asyncio.TimeoutError:
                delay = min(delay * 2, 300.0)
    return None


def start_links(project_id: int, limit: int = 0) -> Dict[str, Any]:
    current = _jobs.get((project_id, "links-memento"))
    if current and current.running:
        return {"status": "already_running"}
    state = JobState(kind="links-memento", running=True, started_at=datetime.utcnow())
    _jobs[(project_id, "links-memento")] = state

    async def runner():
        try:
            await recover_links(project_id, state, limit)
            state.outcome = "stopped" if state.stop.is_set() else "completed"
        except Exception as exc:
            state.outcome = "failed"
            state.message = f"{type(exc).__name__}: {exc}"[:300]
            log.exception("Memento run failed", extra={"project_id": project_id})
        finally:
            state.running = False
            state.current = None
            state.finished_at = datetime.utcnow()

    state.task = asyncio.create_task(runner())
    return {"status": "started"}


def stop_links(project_id: int) -> None:
    state = _jobs.get((project_id, "links-memento"))
    if state and state.running:
        state.stop.set()
