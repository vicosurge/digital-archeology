"""Follow the studied site's links: extract, recover from the Wayback Machine, classify.

A site's own pages only tell part of its story. What a news site cited, which
collaborators and events it promoted, where it sent its readers -- that lives in
its links. This module turns them into material a historian can read:

1. **Extract** (no LLM, no network). Every in-scope capture is parsed and each
   link becomes a ``LinkTarget`` (one per URL) plus a ``PageLink`` per citing
   capture. Captures are all parsed, not one per URL, because a homepage links
   to different articles every day and the capture's timestamp dates the
   citation.

2. **Classify by heuristics** (no LLM). Navigation, tag pages, share buttons
   and newsletter sign-ups are told apart from articles and references, so the
   expensive steps only touch material worth reading.

3. **Recover** (network, no LLM). Selected targets are fetched from the Wayback
   Machine at the capture *closest to when the site first cited them*. The gap
   is recorded (``capture_distance_days``): a capture years away from the
   citation may show a different page than the one the site pointed at.

4. **Analyze** (LLM, metered). Recovered pages are summarized and tagged with
   how they relate to the studied site. This is the only step that spends
   quota, so it is capped per run and stops at a session/weekly usage ceiling.

Recovery asks the Wayback replay endpoint for the capture nearest the citation
date, which also downloads it. The availability and CDX APIs go down and
throttle independently of replay, so they are not on the hot path; throttling
and outages are treated as "try again later", never as "not archived".
"""
import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import aiofiles
import httpx
from sqlalchemy import delete, insert, select, text, update

from ..config import get_llm_config, resolve_local_path
from ..database import DATABASE_DIR, async_session_maker
from ..models import LinkTarget, PageLink, Project
from .llm import (CONTENT_TYPES, SENTIMENTS, _as_str_list, _pick_enum, failure_raw,
                  llm_service, unparseable_error)
from .llm_client import OllamaClient, OllamaError, RateLimiter

import logging as _logging
log = _logging.getLogger("app.links")

LINKED_DIR = os.path.join(DATABASE_DIR, "linked")

# ===================== classification heuristics =====================

MEDIA_EXT = {
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "ico", "tif", "tiff",
    "mp3", "mp4", "m4a", "m4v", "wav", "ogg", "flv", "avi", "mov", "wmv", "swf", "webm",
}
DOCUMENT_EXT = {"pdf", "doc", "docx", "ppt", "pptx", "pps", "xls", "xlsx", "odt", "rtf", "epub"}
DOWNLOAD_EXT = {"zip", "rar", "7z", "gz", "tgz", "exe", "msi", "dmg", "apk", "jar", "iso", "torrent"}

SOCIAL_SITES = {
    "facebook.com", "fb.com", "fb.me", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "youtu.be", "vimeo.com", "flickr.com", "flic.kr", "soundcloud.com",
    "tumblr.com", "pinterest.com", "linkedin.com", "plus.google.com", "myspace.com",
    "foursquare.com", "vine.co", "about.me", "issuu.com", "slideshare.net", "scribd.com",
    "mixcloud.com", "bandcamp.com", "last.fm", "delicious.com", "digg.com", "reddit.com",
    "meneame.net", "stumbleupon.com", "whatsapp.com", "telegram.me",
}
NEWSLETTER_SITES = {"eepurl.com", "mailchi.mp", "feedburner.com", "feedblitz.com", "list-manage.com"}
ADVERTISING_SITES = {
    "doubleclick.net", "googlesyndication.com", "googleadservices.com", "addthis.com",
    "sharethis.com", "addtoany.com", "statcounter.com", "google-analytics.com",
    "histats.com", "clustrmaps.com", "gravatar.com", "wordpress.com/stats",
}

# Paths that list or navigate rather than publish: WordPress/Blogger taxonomy
# and utility URLs in English and Spanish.
TAXONOMY_RE = re.compile(
    r"/(tag|tags|category|categoria|categorias|etiqueta|etiquetas|author|autor|page|pagina|"
    r"feed|rss|atom|comments|trackback|search|buscar|busqueda|attachment|date|archive|"
    r"archivo|wp-login\.php|wp-admin|xmlrpc\.php|wp-json|label|search)(/|$)",
    re.I,
)
TAXONOMY_QUERY_KEYS = {"s", "replytocom", "feed", "share", "cat", "tag", "paged", "author",
                       "attachment_id", "print", "output", "m"}
DATE_ARCHIVE_RE = re.compile(r"^/(\d{4})(/\d{1,2}){0,2}/?$")
ATTACHMENT_SLUG_RE = re.compile(r"^(p|img|dsc|dscn|dscf|imag|image|photo|foto|pic|screenshot)?[_-]?\d{3,}([_-]\d+)*$", re.I)
TRACKING_PARAM_RE = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|_ga|ref|source)$", re.I)
WAYBACK_PREFIX_RE = re.compile(
    r"^(?:https?:)?(?://(?:web\.|wayback\.)?archive\.org)?/web/\d{1,14}[a-z_]*/", re.I
)

# A link on more than this share of the site's distinct pages is chrome
# (menus, footers, partner banners), not something an author chose to cite.
SITEWIDE_SHARE = 0.25

# Parallel Wayback requests during recovery. The Archive throttles bursts, and
# 429s are backed off, so keep this modest.
RECOVER_CONCURRENCY = 3
RECOVER_REQUESTS_PER_MINUTE = 40

# Kinds worth recovering, in the order recovery works through them.
RECOVER_KINDS = ("reference", "article", "document", "sitewide")
KIND_PRIORITY = {k: i for i, k in enumerate(RECOVER_KINDS)}

ANALYZABLE_PREFIXES = ("text/html", "application/xhtml", "text/plain", "application/xml", "text/xml")

RELATIONS = {
    "source_cited",             # a news source, study, or document the site drew on
    "further_reading",          # background the site pointed readers to
    "collaborator_or_partner",  # an allied project, contributor, or partner organization
    "event_or_promotion",       # an event, festival, call or campaign the site promoted
    "own_project",              # another project of the site's own team
    "commercial_or_sponsor",    # an advertiser, sponsor, shop or service provider
    "social_profile",           # a profile page of a person or organization
    "unrelated",
    "other",
}
LINK_CONTENT_TYPES = CONTENT_TYPES | {"unavailable"}


def _site(host: str) -> str:
    return re.sub(r"^www\d*\.", "", host.lower().strip("."))


def _matches_site(site: str, names: Iterable[str]) -> bool:
    return any(site == n or site.endswith("." + n) for n in names)


def _extension(path: str) -> str:
    tail = path.rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[-1].lower() if "." in tail else ""


@dataclass
class NormalizedLink:
    url: str
    key: str
    site: str
    path: str
    query: str


def normalize_link(href: Optional[str], base_url: str) -> Optional[NormalizedLink]:
    """Resolve an href against the page it appeared on, or None if it is not a web link."""
    href = (href or "").strip()
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
        return None
    # Captures stored with id_ keep original links, but some pages were saved
    # through the replay UI and carry /web/<ts>/ prefixes.
    href = WAYBACK_PREFIX_RE.sub("", href)
    href = re.sub(r"^(https?:)/(?!/)", r"\1//", href)   # the Archive collapses '//' in paths
    try:
        parts = urlsplit(urljoin(base_url, href))
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    host = parts.hostname or ""
    if "." not in host or not re.fullmatch(r"[a-z0-9.\-]+", host):
        return None
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not TRACKING_PARAM_RE.match(k)
    ))
    site = _site(host)
    key = site + (f":{port}" if port not in (None, 80, 443) else "") + path.rstrip("/").lower()
    if query:
        key += "?" + query
    return NormalizedLink(
        url=urlunsplit((parts.scheme, netloc, path, query, "")),
        key=key, site=site, path=path, query=query,
    )


def classify_link(link: NormalizedLink, is_internal: bool, share: float) -> str:
    """Heuristic kind for a link. Order matters: file types beat everything."""
    ext = _extension(link.path)
    if ext in MEDIA_EXT:
        return "media"
    if ext in DOCUMENT_EXT:
        return "document"
    if ext in DOWNLOAD_EXT:
        return "download"

    if is_internal:
        stripped = link.path.rstrip("/")
        if stripped in ("", "/index.php", "/index.html") or re.fullmatch(r"/[a-z]{2}", stripped):
            return "navigation"
        if TAXONOMY_RE.search(link.path) or DATE_ARCHIVE_RE.match(link.path):
            return "taxonomy"
        query_keys = {k.lower() for k, _ in parse_qsl(link.query, keep_blank_values=True)}
        if query_keys & TAXONOMY_QUERY_KEYS:
            return "taxonomy"
        if "/wp-content/" in link.path or "/wp-includes/" in link.path:
            return "other"
        # WordPress attachment pages: a photo's own page under its gallery post,
        # named after the camera file (/berlin-las-cicatrices-del-dolor/p1070234/).
        segments = [s for s in link.path.split("/") if s]
        if len(segments) >= 2 and ATTACHMENT_SLUG_RE.match(segments[-1]):
            return "media"
        if share > SITEWIDE_SHARE:
            return "navigation"
        return "article"

    if _matches_site(link.site, SOCIAL_SITES):
        return "social"
    if _matches_site(link.site, NEWSLETTER_SITES) or "list-manage" in link.site:
        return "newsletter"
    if _matches_site(link.site, ADVERTISING_SITES):
        return "advertising"
    if share > SITEWIDE_SHARE:
        return "sitewide"
    return "reference"


class _LinkParser(HTMLParser):
    """Collects (href, anchor text) pairs. Tolerant of the broken HTML old sites are made of."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self.base: Optional[str] = None
        self._open: Optional[Tuple[str, List[str]]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "base" and self.base is None:
            self.base = dict(attrs).get("href")
        elif tag == "area":
            href = dict(attrs).get("href")
            if href:
                self.links.append((href, ""))
        elif tag == "a":
            self._close()
            href = dict(attrs).get("href")
            if href:
                self._open = (href, [])
        elif tag == "img" and self._open is not None:
            alt = dict(attrs).get("alt")
            if alt:
                self._open[1].append(alt)

    def handle_data(self, data):
        if self._open is not None:
            self._open[1].append(data)

    def handle_endtag(self, tag):
        if tag == "a":
            self._close()

    def _close(self):
        if self._open is not None:
            href, parts = self._open
            self._open = None
            self.links.append((href, " ".join(" ".join(parts).split())[:200]))

    def close(self):
        try:
            super().close()
        finally:
            self._close()


def extract_links_from_file(path: str, page_url: str) -> List[Tuple[NormalizedLink, str]]:
    """Parse one capture. Synchronous and CPU-bound: call from a thread."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            html = fh.read()
    except OSError:
        return []
    parser = _LinkParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # HTMLParser can still choke on pathological markup
        pass
    base = page_url
    if parser.base:
        try:
            base = urljoin(page_url, WAYBACK_PREFIX_RE.sub("", parser.base))
        except ValueError:
            pass
    seen = {}
    for href, anchor in parser.links:
        link = normalize_link(href, base)
        if link and (link.key not in seen or (anchor and not seen[link.key][1])):
            seen[link.key] = (link, anchor)
    return list(seen.values())


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts or len(ts) < 8 or not ts[:8].isdigit():
        return None
    try:
        return datetime.strptime(ts[:14].ljust(14, "0"), "%Y%m%d%H%M%S")
    except ValueError:
        try:
            return datetime.strptime(ts[:8], "%Y%m%d")
        except ValueError:
            return None


def wayback_view_url(timestamp: Optional[str], url: str) -> str:
    """Human-facing replay URL, the form a historian cites."""
    return f"https://web.archive.org/web/{timestamp or '*'}/{url}"


# ===================== job bookkeeping =====================

@dataclass
class JobState:
    kind: str
    running: bool = False
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    done: int = 0
    total: int = 0
    errors: int = 0
    calls: int = 0
    current: Optional[str] = None
    message: Optional[str] = None
    outcome: Optional[str] = None
    usage: Optional[float] = None   # latest quota reading, fraction of the allowance


class TransientWaybackError(Exception):
    """The Archive is throttling or offline; the target should be retried later."""


UNCONFIRMED = object()
TIMED_OUT = object()
MAX_TIMEOUT_ATTEMPTS = 3
UNCONFIRMED_NOTE = "unconfirmed: availability API had no capture and CDX was unreachable"


# ===================== the service =====================

class LinkService:
    def __init__(self):
        self._jobs: Dict[Tuple[int, str], JobState] = {}

    # ----- jobs -----
    def job(self, project_id: int, kind: str) -> JobState:
        return self._jobs.get((project_id, kind)) or JobState(kind=kind)

    def _start(self, project_id: int, kind: str, coro_factory) -> Dict[str, Any]:
        current = self._jobs.get((project_id, kind))
        if current and current.running:
            return {"status": "already_running"}
        state = JobState(kind=kind, running=True, started_at=datetime.utcnow())
        self._jobs[(project_id, kind)] = state

        async def runner():
            try:
                await coro_factory(state)
                state.outcome = state.outcome or ("stopped" if state.stop.is_set() else "completed")
            except asyncio.CancelledError:
                state.outcome = "cancelled"
                raise
            except Exception as exc:
                state.outcome = "failed"
                state.message = f"{type(exc).__name__}: {get_llm_config().redact(str(exc))}"
            finally:
                state.running = False
                state.current = None
                state.finished_at = datetime.utcnow()

        state.task = asyncio.create_task(runner())
        return {"status": "started"}

    def stop(self, project_id: int, kind: str) -> Dict[str, Any]:
        state = self._jobs.get((project_id, kind))
        if not state or not state.running:
            return {"status": "not_running"}
        state.stop.set()
        return {"status": "stopping"}

    async def requeue(self, project_id: int, which: str) -> int:
        """Put finished targets back in a queue.

        ``unconfirmed``: "not archived" verdicts reached while CDX was down.
        ``errors``: recovery errors. ``analysis_errors``: failed LLM analyses.
        """
        async with async_session_maker() as session:
            if which == "unconfirmed":
                stmt = (update(LinkTarget)
                        .where(LinkTarget.recovery_status == "not_archived")
                        .where(LinkTarget.error_message == UNCONFIRMED_NOTE)
                        .values(recovery_status="pending"))
            elif which == "errors":
                stmt = (update(LinkTarget).where(LinkTarget.recovery_status == "error")
                        .values(recovery_status="pending"))
            elif which == "analysis_errors":
                stmt = (update(LinkTarget).where(LinkTarget.analysis_status == "error")
                        .values(analysis_status="pending"))
            else:
                raise ValueError(f"Unknown queue: {which}")
            result = await session.execute(stmt.where(LinkTarget.project_id == project_id))
            await session.commit()
            return result.rowcount or 0

    def start_extract(self, project_id: int) -> Dict[str, Any]:
        return self._start(project_id, "extract", lambda s: self.extract(project_id, s))

    def start_recover(self, project_id: int, limit: int = 0) -> Dict[str, Any]:
        return self._start(project_id, "recover", lambda s: self.recover(project_id, s, limit))

    def start_analyze(self, project_id: int, limit: int = 0) -> Dict[str, Any]:
        return self._start(project_id, "analyze", lambda s: self.analyze(project_id, s, limit))

    # ----- 1+2. extraction and heuristic classification -----
    async def extract(self, project_id: int, state: Optional[JobState] = None) -> Dict[str, Any]:
        state = state or JobState(kind="extract")
        async with async_session_maker() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise ValueError("Project not found")
            own_site = _site(project.domain.split("/")[0])

            sources = (await session.execute(text("""
                SELECT id, original_url, timestamp, local_path FROM archived_files
                WHERE project_id = :pid AND in_scope = 1 AND local_path IS NOT NULL
                  AND status IN ('DOWNLOADED', 'PROCESSING', 'PROCESSED', 'SKIPPED', 'ERROR')
                  AND (mimetype IS NULL OR mimetype IN ('text/html', 'application/xhtml+xml', 'unk', ''))
            """), {"pid": project_id})).fetchall()

            # What the corpus already holds, so internal links to captured pages
            # are not "recovered" a second time. Prefer an analyzed capture.
            corpus_rows = (await session.execute(text("""
                SELECT id, original_url, status, timestamp FROM archived_files
                WHERE project_id = :pid AND in_scope = 1 AND local_path IS NOT NULL
                  AND status IN ('DOWNLOADED', 'PROCESSING', 'PROCESSED', 'SKIPPED', 'ERROR')
                ORDER BY CASE status WHEN 'PROCESSED' THEN 0 ELSE 1 END, timestamp DESC
            """), {"pid": project_id})).fetchall()

        corpus: Dict[str, int] = {}
        for fid, url, _status, _ts in corpus_rows:
            link = normalize_link(url, url)
            if link and link.key not in corpus:
                corpus[link.key] = fid

        state.total = len(sources)
        stats: Dict[str, Dict[str, Any]] = {}
        edges: List[Tuple[int, str, str]] = []      # (source_file_id, key, anchor)
        source_keys = {link.key for link in (normalize_link(u, u) for _, u, _, _ in sources) if link}
        source_url_count = len(source_keys) or 1

        def parse_all():
            for fid, url, ts, local_path in sources:
                if state.stop.is_set():
                    return
                state.current = url
                own = normalize_link(url, url)
                for link, anchor in extract_links_from_file(resolve_local_path(local_path), url):
                    if own and link.key == own.key:
                        continue  # self-links carry no information
                    st = stats.get(link.key)
                    if st is None:
                        st = stats[link.key] = {
                            "link": link, "sources": set(), "source_urls": set(),
                            "first": ts, "last": ts, "anchors": Counter(),
                        }
                    st["sources"].add(fid)
                    if own:
                        st["source_urls"].add(own.key)
                    if ts and (not st["first"] or ts < st["first"]):
                        st["first"] = ts
                    if ts and (not st["last"] or ts > st["last"]):
                        st["last"] = ts
                    if anchor:
                        st["anchors"][anchor] += 1
                    edges.append((fid, link.key, anchor))
                state.done += 1

        await asyncio.to_thread(parse_all)
        if state.stop.is_set():
            state.message = "Stopped before writing; nothing was changed."
            return {"status": "stopped"}

        state.current = "Writing links to the database…"
        async with async_session_maker() as session:
            existing = {
                key: (tid, rstatus, astatus)
                for tid, key, rstatus, astatus in (await session.execute(
                    select(LinkTarget.id, LinkTarget.url_key, LinkTarget.recovery_status,
                           LinkTarget.analysis_status)
                    .where(LinkTarget.project_id == project_id)
                )).all()
            }
            new_rows, updates = [], []
            for key, st in stats.items():
                link: NormalizedLink = st["link"]
                is_internal = link.site == own_site or link.site.endswith("." + own_site)
                share = len(st["source_urls"]) / source_url_count
                kind = classify_link(link, is_internal, share)
                corpus_id = corpus.get(key) if is_internal else None
                anchor = st["anchors"].most_common(1)[0][0] if st["anchors"] else None
                values = {
                    "url": link.url, "domain": link.site, "is_internal": is_internal,
                    "kind": kind, "corpus_file_id": corpus_id,
                    "cited_by_pages": len(st["sources"]),
                    "first_cited_ts": st["first"], "last_cited_ts": st["last"],
                    "sample_anchor": anchor,
                }
                prior = existing.get(key)
                prior_status = prior[1] if prior else None
                if corpus_id:
                    status = "in_corpus"
                elif kind in RECOVER_KINDS:
                    # Keep a finished outcome; only (re)queue what was never tried.
                    status = prior_status if prior_status in ("recovered", "not_archived", "error", "pending") \
                        else "pending"
                else:
                    status = prior_status if prior_status == "recovered" else "not_selected"
                values["recovery_status"] = status
                if prior:
                    updates.append({"id": prior[0], **values})
                else:
                    new_rows.append({"project_id": project_id, "url_key": key,
                                     "analysis_status": "not_selected", **values})

            for chunk in _chunks(new_rows, 2000):
                await session.execute(insert(LinkTarget), chunk)
            for chunk in _chunks(updates, 2000):
                await session.execute(update(LinkTarget), chunk)
            # Targets no longer linked from any in-scope capture (the scope
            # window changed) keep their row but drop out of recovery.
            stale = [tid for key, (tid, rstatus, _a) in existing.items()
                     if key not in stats and rstatus == "pending"]
            for chunk in _chunks(stale, 900):
                await session.execute(
                    update(LinkTarget).where(LinkTarget.id.in_(chunk))
                    .values(recovery_status="not_selected", cited_by_pages=0)
                )
            await session.flush()

            ids = dict((await session.execute(
                select(LinkTarget.url_key, LinkTarget.id).where(LinkTarget.project_id == project_id)
            )).all())
            await session.execute(delete(PageLink).where(PageLink.project_id == project_id))
            edge_rows = [
                {"project_id": project_id, "source_file_id": fid, "target_id": ids[key],
                 "anchor_text": anchor or None}
                for fid, key, anchor in edges if key in ids
            ]
            for chunk in _chunks(edge_rows, 5000):
                await session.execute(insert(PageLink), chunk)
            await session.commit()

        kinds = Counter()
        for key, st in stats.items():
            link = st["link"]
            is_internal = link.site == own_site or link.site.endswith("." + own_site)
            kinds[("internal" if is_internal else "external")] += 1
        state.message = (f"Parsed {state.done} captures: {len(stats)} distinct links "
                         f"({kinds['internal']} internal, {kinds['external']} external), "
                         f"{len(edge_rows)} citations.")
        return {"status": "completed", "targets": len(stats), "citations": len(edge_rows)}

    # ----- 3. recovery from the Wayback Machine -----
    async def recover(self, project_id: int, state: Optional[JobState] = None,
                      limit: int = 0, concurrency: int = RECOVER_CONCURRENCY) -> Dict[str, Any]:
        """Work through pending targets with a pool of workers.

        Workers never touch the ORM: they take a plain snapshot of a target and
        hand back the column values to write, and this loop alone owns the
        session. A capture that replays slowly occupies one worker instead of
        stalling the others. Throttling (429/5xx) pauses the whole pool with a
        growing back-off, since it is the Archive telling every worker the same
        thing.
        """
        state = state or JobState(kind="recover")
        out_dir = os.path.join(LINKED_DIR, str(project_id))
        os.makedirs(out_dir, exist_ok=True)

        async with async_session_maker() as session:
            state.total = (await session.execute(text(
                "SELECT COUNT(*) FROM link_targets WHERE project_id = :pid AND recovery_status = 'pending'"
            ), {"pid": project_id})).scalar() or 0
        if limit:
            state.total = min(state.total, limit)

        work_q: asyncio.Queue = asyncio.Queue(maxsize=concurrency)
        results_q: asyncio.Queue = asyncio.Queue()
        gate = asyncio.Event()          # cleared while backing off
        gate.set()
        in_flight: set = set()
        # One pace for the whole pool: the Archive limits by client, not by worker.
        pace = RateLimiter(RECOVER_REQUESTS_PER_MINUTE)
        headers = {"User-Agent": "DigitalArcheology/1.0 (historical research; link recovery)"}

        async def worker(client: httpx.AsyncClient):
            while True:
                snap = await work_q.get()
                try:
                    await gate.wait()
                    try:
                        values = await self._recover_one(client, snap, out_dir, pace)
                        await results_q.put((snap, values, None))
                    except TransientWaybackError as exc:
                        await results_q.put((snap, None, exc))
                    except Exception as exc:
                        await results_q.put((snap, {"recovery_status": "error",
                                                    "error_message": f"{type(exc).__name__}: {exc}"[:500]},
                                             None))
                finally:
                    work_q.task_done()

        consecutive_transient = 0
        backoff = 30.0
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0),
                                     follow_redirects=True, headers=headers,
                                     limits=httpx.Limits(max_connections=concurrency * 2)) as client:
            workers = [asyncio.create_task(worker(client)) for _ in range(concurrency)]
            try:
                async with async_session_maker() as session:
                    exhausted = False
                    while True:
                        # Top up the queue.
                        if not exhausted and not state.stop.is_set() and work_q.qsize() < concurrency \
                                and (not limit or state.done + len(in_flight) < limit):
                            rows = (await session.execute(
                                select(LinkTarget.id, LinkTarget.url, LinkTarget.first_cited_ts,
                                       LinkTarget.attempts, LinkTarget.analysis_status)
                                .where(LinkTarget.project_id == project_id)
                                .where(LinkTarget.recovery_status == "pending")
                                .where(LinkTarget.id.notin_(in_flight) if in_flight else text("1=1"))
                                .order_by(
                                    LinkTarget.attempts,
                                    text("CASE kind WHEN 'reference' THEN 0 WHEN 'article' THEN 1 "
                                         "WHEN 'document' THEN 2 ELSE 3 END"),
                                    LinkTarget.cited_by_pages.desc(), LinkTarget.id,
                                )
                                .limit(concurrency * 2)
                            )).all()
                            if not rows and not in_flight:
                                exhausted = True
                            for row in rows:
                                if limit and state.done + len(in_flight) >= limit:
                                    break
                                snap = {"id": row[0], "url": row[1], "first_cited_ts": row[2],
                                        "attempts": row[3] or 0, "analysis_status": row[4]}
                                in_flight.add(snap["id"])
                                await work_q.put(snap)

                        if not in_flight and (exhausted or state.stop.is_set()
                                              or (limit and state.done >= limit)):
                            break

                        try:
                            snap, values, transient = await asyncio.wait_for(results_q.get(), timeout=2.0)
                        except asyncio.TimeoutError:
                            continue
                        in_flight.discard(snap["id"])
                        state.current = snap["url"]

                        if transient is not None:
                            await session.execute(update(LinkTarget).where(LinkTarget.id == snap["id"])
                                                  .values(error_message=str(transient)[:500]))
                            await session.commit()
                            consecutive_transient += 1
                            if consecutive_transient >= 8:
                                state.outcome = "paused"
                                state.message = ("The Wayback Machine keeps refusing requests "
                                                 f"({transient}). Stopped; press Recover to resume later.")
                                state.stop.set()
                                continue
                            if gate.is_set():
                                gate.clear()
                                state.message = f"Wayback busy ({transient}); waiting {int(backoff)}s"
                                await _sleep_unless(state.stop, backoff)
                                backoff = min(backoff * 2, 300.0)
                                state.message = None
                                gate.set()
                            continue

                        consecutive_transient = 0
                        backoff = 30.0
                        await session.execute(update(LinkTarget).where(LinkTarget.id == snap["id"])
                                              .values(**values))
                        await session.commit()
                        if values.get("recovery_status") != "pending":
                            state.done += 1
                        if values.get("recovery_status") == "error":
                            state.errors += 1
                            log.warning("Linked page recovery failed: %s: %s", snap["url"],
                                        values.get("error_message"), extra={"project_id": project_id})
            finally:
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
        return {"status": state.outcome or "completed", "done": state.done}

    async def _recover_one(self, client: httpx.AsyncClient, snap: Dict[str, Any],
                           out_dir: str, pace: Optional[RateLimiter] = None) -> Dict[str, Any]:
        """Recover one target. Returns the column values to write; raises TransientWaybackError.

        Asks the replay endpoint directly for ``/web/<cited timestamp>id_/<url>``:
        the Wayback Machine redirects that to the capture closest to the
        timestamp and serves it, or answers 404 when the URL was never archived.
        One endpoint does both the lookup and the download, and it is the one
        the Archive throttles least -- the availability API started answering
        429 within minutes of steady use.
        """
        attempts = snap["attempts"] + 1
        cited_ts = (snap["first_cited_ts"] or "")[:14] or "2000"
        url = snap["url"]
        if pace:
            await pace.acquire()
        try:
            response = await client.get(f"https://web.archive.org/web/{cited_ts}id_/{url}")
        except httpx.TimeoutException:
            return _retry_or_fail(attempts, "Timed out fetching the capture")
        except httpx.TransportError as exc:
            raise TransientWaybackError(f"{type(exc).__name__} fetching capture") from exc
        if response.status_code in (429, 502, 503, 504):
            raise TransientWaybackError(f"HTTP {response.status_code} fetching capture")
        if response.status_code == 404:
            return {"recovery_status": "not_archived", "attempts": attempts, "error_message": None}
        if response.status_code >= 400:
            return {"recovery_status": "error", "attempts": attempts,
                    "error_message": f"HTTP {response.status_code} fetching capture"}

        content = response.content
        if len(content) > 15_000_000:
            return {"recovery_status": "error", "attempts": attempts,
                    "error_message": f"Capture too large ({len(content)} bytes)"}
        mimetype = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        final_url = str(response.url)
        final_ts = _timestamp_from_wayback_url(final_url)
        original = _original_from_wayback_url(final_url) or url
        if not final_ts:
            return {"recovery_status": "error", "attempts": attempts,
                    "error_message": f"Replay did not land on a capture ({final_url[:200]})"}
        path = os.path.join(out_dir, f"{snap['id']}{_file_extension(mimetype, original)}")
        async with aiofiles.open(path, "wb") as fh:
            await fh.write(content)

        cited, captured = _parse_ts(snap["first_cited_ts"]), _parse_ts(final_ts)
        values = {
            "recovery_status": "recovered", "attempts": attempts, "error_message": None,
            "wayback_timestamp": final_ts, "wayback_url": wayback_view_url(final_ts, original),
            "capture_distance_days": abs((captured - cited).days) if cited and captured else None,
            "mimetype": mimetype or None, "local_path": path, "file_size": len(content),
            "recovered_at": datetime.utcnow(),
        }
        if mimetype.startswith(ANALYZABLE_PREFIXES) or not mimetype:
            if snap["analysis_status"] in (None, "not_selected"):
                values["analysis_status"] = "pending"
        else:
            values["analysis_status"] = "skipped"
        return values

    async def _closest_capture(self, client: httpx.AsyncClient, url: str, timestamp: str):
        """(timestamp, original url) of the capture nearest ``timestamp``.

        None when the Archive has no capture, UNCONFIRMED when only the
        availability API could be asked and it had none.
        """
        availability_unsure = False
        try:
            response = await client.get(
                "https://archive.org/wayback/available",
                params={"url": url, "timestamp": timestamp[:14] or "2000"},
            )
            if response.status_code in (429, 502, 503, 504):
                raise TransientWaybackError(f"availability API HTTP {response.status_code}")
            data = response.json()
            closest = ((data or {}).get("archived_snapshots") or {}).get("closest") or {}
            if closest.get("available") and closest.get("timestamp"):
                original = _original_from_wayback_url(closest.get("url") or "") or url
                return closest["timestamp"], original
        except (httpx.TimeoutException, httpx.TransportError, ValueError):
            availability_unsure = True

        # The availability API is known to miss captures, especially for URLs
        # with query strings. Ask the CDX index before concluding there is none.
        try:
            response = await client.get(
                "https://web.archive.org/cdx/search/cdx",
                params={"url": url, "closest": timestamp[:14] or "2000", "sort": "closest",
                        "limit": "3", "output": "json", "fl": "timestamp,original,statuscode",
                        "filter": "!statuscode:[45]..", "matchType": "exact"},
            )
            if response.status_code >= 500 or response.status_code == 429:
                if availability_unsure:
                    raise TransientWaybackError(f"availability unreachable and CDX HTTP {response.status_code}")
                return UNCONFIRMED
            rows = response.json()
        except (httpx.TimeoutException, httpx.TransportError, ValueError):
            # CDX serves an HTML "Temporarily Offline" page when it is down.
            if availability_unsure:
                raise TransientWaybackError("availability and CDX APIs both unreachable")
            # The availability API did answer "no capture". Accept it, but mark
            # the verdict unconfirmed so recheck_unconfirmed() can revisit it
            # once CDX is back, instead of stalling the whole run on CDX.
            return UNCONFIRMED
        if isinstance(rows, list) and len(rows) > 1:
            header, best = rows[0], rows[1]
            record = dict(zip(header, best))
            return record["timestamp"], record["original"]
        return None

    # ----- 4. LLM analysis of recovered pages -----
    async def analyze(self, project_id: int, state: Optional[JobState] = None,
                      limit: int = 0) -> Dict[str, Any]:
        state = state or JobState(kind="analyze")
        cfg = get_llm_config()
        max_calls = limit or cfg.max_calls_per_run or 0

        async with async_session_maker() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise ValueError("Project not found")
            model = project.ollama_model or cfg.model
            state.total = (await session.execute(text(
                "SELECT COUNT(*) FROM link_targets WHERE project_id = :pid AND analysis_status = 'pending'"
            ), {"pid": project_id})).scalar() or 0
        if max_calls:
            state.total = min(state.total, max_calls)

        health = await llm_service.check_ollama(cfg)
        if health.get("status") != "running":
            state.outcome = "failed"
            state.message = f"LLM endpoint not reachable: {health.get('message', 'unknown error')}"
            return {"status": "error"}
        available = health.get("models") or []
        if available and model not in available:
            model = cfg.model if cfg.model in available else available[0]

        usage_due = 0
        async with OllamaClient(cfg, RateLimiter(cfg.requests_per_minute)) as client:
            while not state.stop.is_set():
                room = cfg.concurrency * 2
                if max_calls:
                    room = min(room, max_calls - state.calls)
                    if room <= 0:
                        state.outcome = "budget_reached"
                        state.message = f"Reached the limit of {max_calls} LLM calls for this run."
                        break
                if usage_due <= 0:
                    usage_due = 25
                    usage = await client.fetch_usage() or {}
                    ceiling = cfg.stop_at_session_usage or 0.95
                    if usage.get("max", 0) >= ceiling:
                        state.outcome = "budget_reached"
                        state.message = (f"Quota ceiling reached ({usage['max'] * 100:.0f}% of the "
                                         f"endpoint's allowance, limit {ceiling * 100:.0f}%). Progress is saved.")
                        break
                    state.usage = usage.get("max")

                async with async_session_maker() as session:
                    batch = (await session.execute(
                        select(LinkTarget)
                        .where(LinkTarget.project_id == project_id)
                        .where(LinkTarget.analysis_status == "pending")
                        .order_by(LinkTarget.cited_by_pages.desc(), LinkTarget.id)
                        .limit(room)
                    )).scalars().all()
                    if not batch:
                        break
                    contexts = await self._citation_context(session, [t.id for t in batch])
                    state.current = batch[0].url
                    results = await asyncio.gather(*[
                        self._analyze_one(client, cfg, model, project, t, contexts.get(t.id, []))
                        for t in batch
                    ])
                    fatal = None
                    for target, (outcome, data, raw, error, is_fatal, called) in zip(batch, results):
                        if called:
                            state.calls += 1
                            usage_due -= 1
                        state.done += 1
                        target.analyzed_at = datetime.utcnow()
                        target.model_used = model
                        target.raw_response = raw
                        if outcome == "processed":
                            target.analysis_status = "processed"
                            for k, v in data.items():
                                setattr(target, k, v)
                            target.error_message = None
                        elif outcome == "skipped":
                            target.analysis_status = "skipped"
                            target.error_message = error
                        else:
                            target.analysis_status = "error"
                            target.error_message = (error or "")[:500]
                            state.errors += 1
                            log.warning("Linked page analysis failed: %s: %s", target.url, error,
                                        extra={"project_id": project_id})
                            if is_fatal:
                                fatal = error
                    await session.commit()
                    if fatal:
                        state.outcome = "failed"
                        state.message = fatal
                        break
        return {"status": state.outcome or "completed", "calls": state.calls}

    async def _citation_context(self, session, target_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Up to three citing pages per target, with the anchor text and that page's summary."""
        if not target_ids:
            return {}
        placeholders = ",".join(str(int(i)) for i in target_ids)
        rows = (await session.execute(text(f"""
            SELECT pl.target_id, pl.anchor_text, af.original_url, af.timestamp, fa.description
            FROM page_links pl
            JOIN archived_files af ON af.id = pl.source_file_id
            LEFT JOIN file_analyses fa ON fa.file_id = af.id
            WHERE pl.target_id IN ({placeholders})
            ORDER BY pl.target_id, (fa.description IS NULL), af.timestamp
        """))).fetchall()
        out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        seen_urls: Dict[int, set] = defaultdict(set)
        for tid, anchor, url, ts, desc in rows:
            if len(out[tid]) >= 3 or url in seen_urls[tid]:
                continue
            seen_urls[tid].add(url)
            out[tid].append({"anchor": anchor, "url": url, "timestamp": ts, "description": desc})
        return out

    async def _analyze_one(self, client: OllamaClient, cfg, model: str, project: Project,
                           target: LinkTarget, citations: List[Dict[str, Any]]):
        """Returns (outcome, data, raw, error, fatal, called_llm)."""
        path = resolve_local_path(target.local_path)
        if not path or not os.path.exists(path):
            return "error", None, None, "Recovered file missing from disk", False, False
        content = await asyncio.to_thread(llm_service.read_and_clean, path, cfg.max_content_chars)
        if len(content) < 80:
            return "skipped", None, None, "Too little text to analyze", False, False

        prompt = build_link_prompt(project, target, citations, content, cfg.max_content_chars)
        num_predict = max(cfg.num_predict, 700)
        result: Dict[str, Any] = {}
        for temperature in (0.3, 0.1):
            try:
                result = await client.generate(
                    prompt, model=model, fmt="json",
                    options={"num_predict": num_predict, "temperature": temperature,
                             "top_p": 0.9},
                )
            except OllamaError as exc:
                return "error", None, None, cfg.redact(str(exc)), exc.fatal, True
            raw = result.get("response", "") or ""
            data = validate_link_analysis(llm_service._extract_json(raw))
            if data is not None:
                return "processed", data, raw, None, False, True
            if result.get("done_reason") == "length":
                break   # a retry would be cut off at the same limit
        return ("error", None, failure_raw(result),
                unparseable_error(result, num_predict), False, True)


def build_link_prompt(project: Project, target: LinkTarget, citations: List[Dict[str, Any]],
                      content: str, max_chars: int) -> str:
    cited_lines = []
    for c in citations:
        line = f"- Page {c['url']} (captured {_fmt_ts(c['timestamp'])})"
        if c.get("anchor"):
            line += f", link text: \"{c['anchor']}\""
        if c.get("description"):
            line += f"\n  Summary of that page: {c['description'][:400]}"
        cited_lines.append(line)
    scope = (project.scope_reason or "").strip()
    return f"""You are helping a historian study the website {project.domain}.
{('Context about the site: ' + scope) if scope else ''}

The site linked to the page below. It was recovered from the Internet Archive as it
looked on {_fmt_ts(target.wayback_timestamp)} (the site first linked to it on {_fmt_ts(target.first_cited_ts)}).

Linked URL: {target.url}
Where {project.domain} linked to it:
{chr(10).join(cited_lines) or '- (citing page details unavailable)'}

Content of the linked page:
{content[:max_chars]}

Respond ONLY with a JSON object in this format:
{{
    "title": "the page's title or a short name for it",
    "description": "2-4 sentence summary of what this linked page contains",
    "content_type": "news/economy/discussion/technology/entertainment/personal/business/education/other/unavailable",
    "sentiment": "positive/negative/neutral/mixed",
    "sentiment_score": 0.0,
    "subjects": ["main topics"],
    "tags": ["tags that help find this later"],
    "people_mentioned": ["names of people clearly identified"],
    "relation": "source_cited/further_reading/collaborator_or_partner/event_or_promotion/own_project/commercial_or_sponsor/social_profile/unrelated/other",
    "relation_note": "one sentence on why {project.domain} linked here, grounded in the citing context"
}}

Notes:
- Use content_type "unavailable" if the page is an error page, a parked or for-sale domain,
  a login wall, or has no real content.
- "content_type", "sentiment" and "relation" must be one of the English options exactly as written.
- Write "title", "description", "subjects", "tags" and "relation_note" in the language of the linked page.
- Do not invent information that is not in the content or the citing context."""


def validate_link_analysis(data: Any) -> Optional[Dict[str, Any]]:
    base = llm_service.validate_analysis(data)
    if base is None:
        return None
    title = str(data.get("title") or "").strip()[:300] or None
    relation_note = str(data.get("relation_note") or "").strip()[:1000] or None
    return {
        "title": title,
        "description": base["description"],
        "content_type": _pick_enum(data.get("content_type"), LINK_CONTENT_TYPES, "other"),
        "sentiment": base["sentiment"],
        "sentiment_score": base["sentiment_score"],
        "subjects": base["subjects"],
        "tags": base["tags"],
        "people_mentioned": base["people_mentioned"],
        "relation": _pick_enum(data.get("relation"), RELATIONS, "other"),
        "relation_note": relation_note,
    }


# ===================== small helpers =====================

def _retry_or_fail(attempts: int, message: str) -> Dict[str, Any]:
    """A slow capture is retried on a later pass, then given up on."""
    if attempts >= MAX_TIMEOUT_ATTEMPTS:
        return {"recovery_status": "error", "attempts": attempts, "error_message": message}
    return {"recovery_status": "pending", "attempts": attempts, "error_message": message}


def _chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def _sleep_unless(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts or len(ts) < 8:
        return "an unknown date"
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"


_WAYBACK_URL_RE = re.compile(r"/web/(\d{8,14})[a-z_]*/(.+)$", re.I)


def _timestamp_from_wayback_url(url: str) -> Optional[str]:
    match = _WAYBACK_URL_RE.search(url or "")
    return match.group(1) if match else None


def _original_from_wayback_url(url: str) -> Optional[str]:
    match = _WAYBACK_URL_RE.search(url or "")
    if not match:
        return None
    original = match.group(2)
    return re.sub(r"^(https?:)/(?!/)", r"\1//", original)


def _file_extension(mimetype: str, url: str) -> str:
    by_type = {
        "text/html": ".html", "application/xhtml+xml": ".html", "text/plain": ".txt",
        "application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "application/xml": ".xml", "text/xml": ".xml", "application/rss+xml": ".xml",
    }
    if mimetype in by_type:
        return by_type[mimetype]
    ext = _extension(urlsplit(url).path)
    return f".{ext}" if ext and len(ext) <= 5 and ext.isalnum() else ".bin"


link_service = LinkService()
