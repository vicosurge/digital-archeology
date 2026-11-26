"""Full-text search over the archived pages themselves.

The results page searches what the model *wrote about* each page. This
searches what the pages *say*, including pages that were never analyzed. It
uses SQLite's FTS5 with diacritics folded, so "cancion" finds "canción".

The index is a virtual table whose rowid is the capture's archived_files id, so
updating it is incremental: only captures without a row are read. It is filled
after each download, after each scope change, and from the results page or the
CLI (search-index).

Only in-scope captures are indexed. On a domain that was taken over, the
squatter's pages are most of the text (96% on b1mas4, 240 MB of pharmacy
spam), and searching them is not what the scope is for.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ..database import async_session_maker

log = logging.getLogger("app.search")

# Text kept per page. Large enough for any article; the index is a fraction
# of the corpus because markup is dropped.
MAX_CHARS = 200_000
BATCH = 100
MATCH_CAP = 5000
_TEXTUAL = ("text/html", "text/plain", "application/xhtml+xml")

CREATE_SQL = ("CREATE VIRTUAL TABLE IF NOT EXISTS page_text USING fts5("
              "url, body, project_id UNINDEXED, "
              "tokenize = 'unicode61 remove_diacritics 2')")

_running: Dict[int, asyncio.Task] = {}


def fts_query(user_query: str) -> str:
    """Turn what a person types into a safe FTS5 query.

    Words are quoted so punctuation never becomes syntax ("C++", "e-mail").
    "Quoted phrases" stay phrases, OR and NOT keep their meaning, and a
    trailing * keeps prefix search. Everything else is ANDed, like a search box.
    """
    parts: List[str] = []
    for m in re.finditer(r'"([^"]*)"|(\S+)', user_query or ""):
        phrase, word = m.group(1), m.group(2)
        if phrase is not None:
            words = phrase.replace('"', " ").split()
            if words:
                parts.append('"' + " ".join(words) + '"')
            continue
        if word in ("OR", "NOT", "AND"):
            parts.append(word)
            continue
        prefix = word.endswith("*") and len(word) > 1
        core = word.rstrip("*").replace('"', "")
        if not core:
            continue
        parts.append(f'"{core}"' + ("*" if prefix else ""))
    # A dangling operator is a syntax error.
    while parts and parts[0] in ("OR", "AND", "NOT"):
        parts.pop(0)
    while parts and parts[-1] in ("OR", "AND", "NOT"):
        parts.pop()
    return " ".join(parts)


def url_key(url: str) -> str:
    """One key for the spellings of the same page: scheme, www., :80 and a
    trailing slash are how the archive happened to record it, not different pages."""
    u = re.sub(r"^https?://", "", (url or "").strip().lower())
    u = re.sub(r"^www\.", "", u)
    u = u.replace(":80/", "/", 1)
    return u.rstrip("/")


def render_snippet(raw: str) -> str:
    """HTML-escape a snippet whose matches are wrapped in \\x02 ... \\x03."""
    return html.escape(raw or "").replace("\x02", "<mark>").replace("\x03", "</mark>")


async def pending_count(db, project_id: int) -> int:
    return (await db.execute(text(f"""
        SELECT COUNT(*) FROM archived_files af
        WHERE af.project_id = :pid AND af.local_path IS NOT NULL AND af.in_scope = 1
          AND (af.mimetype IN ({_mime_list()}) OR af.mimetype IS NULL)
          AND NOT EXISTS (SELECT 1 FROM page_text WHERE rowid = af.id)
    """), {"pid": project_id})).scalar() or 0


def _mime_list() -> str:
    return ", ".join(f"'{m}'" for m in _TEXTUAL)


def _read(path: str) -> str:
    from .llm import llm_service
    try:
        return llm_service.read_and_clean(path, MAX_CHARS)
    except OSError:
        return ""


async def prune(db, project_id: int) -> None:
    """Remove captures that are no longer in scope."""
    result = await db.execute(text(
        "DELETE FROM page_text WHERE rowid IN "
        "(SELECT id FROM archived_files WHERE project_id = :pid AND in_scope = 0)"),
        {"pid": project_id})
    if result.rowcount:
        # FTS5 keeps deleted entries in its segments until they are merged.
        await db.execute(text("INSERT INTO page_text(page_text) VALUES ('optimize')"))


async def index_project(project_id: int) -> Dict[str, int]:
    """Index every held in-scope text capture of the project not indexed yet."""
    added = 0
    async with async_session_maker() as db:
        await prune(db, project_id)
        await db.commit()
    while True:
        async with async_session_maker() as db:
            rows = (await db.execute(text(f"""
                SELECT af.id, af.original_url, af.local_path FROM archived_files af
                WHERE af.project_id = :pid AND af.local_path IS NOT NULL AND af.in_scope = 1
                  AND (af.mimetype IN ({_mime_list()}) OR af.mimetype IS NULL)
                  AND NOT EXISTS (SELECT 1 FROM page_text WHERE rowid = af.id)
                ORDER BY af.id LIMIT :n
            """), {"pid": project_id, "n": BATCH})).fetchall()
            if not rows:
                break
            texts = await asyncio.gather(*(asyncio.to_thread(_read, r[2]) for r in rows))
            await db.execute(
                text("INSERT INTO page_text (rowid, url, body, project_id) VALUES (:id, :url, :body, :pid)"),
                [{"id": r[0], "url": r[1], "body": t, "pid": project_id} for r, t in zip(rows, texts)],
            )
            await db.commit()
            added += len(rows)
    if added:
        log.info("Search index: added %d pages", added, extra={"project_id": project_id})
    return {"indexed": added}


def start_index(project_id: int) -> bool:
    """Index in the background unless a pass for this project is running."""
    task = _running.get(project_id)
    if task and not task.done():
        return False

    async def run():
        try:
            await index_project(project_id)
        except Exception:
            log.exception("Search indexing failed", extra={"project_id": project_id})

    _running[project_id] = asyncio.create_task(run())
    return True


def is_indexing(project_id: int) -> bool:
    task = _running.get(project_id)
    return bool(task and not task.done())


async def delete_project(db, project_id: int) -> None:
    await db.execute(text("DELETE FROM page_text WHERE project_id = :pid"), {"pid": project_id})


async def search(db, project_id: int, user_query: str,
                 limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """Matching pages, one row per URL (its latest matching capture)."""
    q = fts_query(user_query)
    if not q:
        return {"query": q, "total": 0, "rows": []}
    base = """
        FROM page_text pt JOIN archived_files af ON af.id = pt.rowid
        WHERE page_text MATCH :q AND pt.project_id = :pid AND af.in_scope = 1
    """
    params = {"q": q, "pid": project_id}
    # bm25() cannot be aggregated in SQL, so rank captures there and collapse
    # them to one row per URL here. MATCH_CAP bounds the work on huge corpora.
    matches = (await db.execute(text(f"""
        SELECT af.original_url, af.timestamp, af.id {base}
        ORDER BY bm25(page_text) LIMIT :cap
    """), {**params, "cap": MATCH_CAP})).fetchall()
    by_url: Dict[str, Dict[str, Any]] = {}
    for url, ts, file_id in matches:
        entry = by_url.setdefault(url_key(url), {"url": url, "timestamp": ts, "file_id": file_id,
                                                 "captures": 0})
        entry["captures"] += 1
        if (ts or "") > (entry["timestamp"] or ""):
            entry.update(url=url, timestamp=ts, file_id=file_id)
    ranked = list(by_url.values())
    total = len(ranked)
    out: List[Dict[str, Any]] = []
    for entry in ranked[offset:offset + limit]:
        snip = (await db.execute(text(
            "SELECT snippet(page_text, 1, char(2), char(3), '…', 28) FROM page_text "
            "WHERE page_text MATCH :q AND rowid = :id"), {"q": q, "id": entry["file_id"]})).scalar()
        out.append({**entry, "snippet": render_snippet(snip or "")})
    return {"query": q, "total": int(total), "rows": out}
