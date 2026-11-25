"""Close reading: the historian's apparatus, one article at a time.

The triage pass (services/llm.py + processing.py) answers "what is this page
and what is it broadly about" for all 1,437 in-scope captures. That is
heuristics -- finding and sorting the sources. This module is the next step of
the method: internal criticism of the individual document.

For each article-shaped page it asks the questions a source is actually
interrogated with:

  who        people, separated by the role the text gives them -- who writes,
             who is quoted, who is written about. The distinction is the whole
             point: a publication that quotes only officials and writes *about*
             residents is a different publication from one that does the
             reverse, and neither will ever say so about itself.
  where      places, typed, so coverage can be mapped rather than asserted.
  when       two clocks. When the page was published, and when the events it
             reports happened. Conflating them is the classic error with
             digital sources, where the capture date is mistaken for both.
  what       the assertions the page makes, in its own words, and the direct
             speech it prints.
  why        genre, intent and stance -- what the text is *for*. A nota
             informativa, a cronica and an editorial about the same event are
             three different acts.

Everything here is descriptive. The model is asked what the page says and how
it says it, never whether it is true; verification is the historian's job and
the archive exists so they can do it.

Cost and safety: this runs only on article-shaped URLs (758 of b1mas4's
in-scope pages, against 665 tag listings and paginations that have no author,
no place and no claim), writes to its own table, and never touches
file_analyses -- re-running it can never endanger the corpus-wide triage.
"""
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, text, update

from ..config import LLMConfig, get_llm_config
from ..database import async_session_maker
from ..models import PageReading, Project
from .links import JobState, classify_link, normalize_link
from .llm import _as_str_list, _pick_enum, failure_raw, llm_service, unparseable_error
from .llm_client import OllamaClient, OllamaError, RateLimiter

import logging as _logging
log = _logging.getLogger("app.reading")

# ===================== vocabularies =====================
#
# Closed vocabularies, in English, for the same reason the triage enums are:
# they are filtered and counted, and a free-text label in three languages
# cannot be. The free text (names, places, summaries, quotes) stays in the
# language of the source.

# Journalistic and literary form. Kept in the Spanish-language terms the
# tradition actually uses where English has no equivalent that means the same
# thing -- a *cronica* is not a "chronicle" and a *reportaje* is not a "report".
GENRES = {
    "nota_informativa",   # straight news
    "reportaje",          # long-form investigative piece
    "cronica",            # narrative, first-person-adjacent account
    "entrevista",
    "columna",            # signed recurring opinion
    "editorial",          # unsigned institutional position
    "resena",             # review of a book, film, show
    "ensayo",
    "literary",           # poem, short story, fragment
    "institutional",      # about the publication itself: staff, mission, notice
    "listing",            # index, agenda, directory
    "other",
}

# What the text is for. This is the "is it political, cultural..." axis.
INTENTS = {
    "political", "cultural", "economic", "social", "educational",
    "literary", "institutional", "commercial", "personal", "other",
}

# Posture toward the subject.
STANCES = {
    "informative",    # reports without taking a position
    "analytical",     # explains, weighs causes
    "critical",       # finds fault, argues against
    "denunciatory",   # accuses, names wrongdoing
    "supportive",     # argues for, defends
    "promotional",    # advertises an event, product or person
    "commemorative",  # honours, remembers, mourns
}

PERSON_ROLES = {"author", "quoted", "subject", "mentioned"}
PLACE_KINDS = {"city", "state", "country", "region", "neighborhood", "venue", "other"}
ORG_KINDS = {"government", "party", "university", "school", "company", "media",
             "ngo", "religious", "union", "cultural", "other"}
TEMPORAL_MODES = {"current_event", "retrospective", "forecast", "timeless"}

# A close reading is a much longer answer than a triage analysis; the shared
# default (500) truncates it mid-JSON, and so did 1600 on the densest pieces --
# a long interview names enough people, places and quotations to run past it.
# The prompt also caps each list, which keeps the answer bounded from the other
# side: a page with forty names is a directory, not an article.
READING_NUM_PREDICT = 2600
READING_MAX_CHARS = 12000


# ===================== prompt =====================

def build_reading_prompt(content: str, url: str, timestamp: Optional[str],
                         max_chars: int = READING_MAX_CHARS) -> str:
    captured = "unknown"
    if timestamp and len(timestamp) >= 8:
        captured = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    return f"""You are helping a historian read one page from an archived website.
Describe what the page says and how it says it. Do not judge whether its claims
are true, and do not add anything the page does not contain.

Original URL: {url}
Date this copy was captured by the web archive: {captured}

Page content:
{content[:max_chars]}

Answer with this JSON object and nothing else:

{{
  "title": "the headline or title of the piece",
  "genre": "nota_informativa|reportaje|cronica|entrevista|columna|editorial|resena|ensayo|literary|institutional|listing|other",
  "intent": "political|cultural|economic|social|educational|literary|institutional|commercial|personal|other",
  "secondary_intents": ["other applicable values from the intent list"],
  "stance": "informative|analytical|critical|denunciatory|supportive|promotional|commemorative",
  "people": [
    {{"name": "full name as written",
      "role": "author|quoted|subject|mentioned",
      "affiliation": "their office, outlet or institution if the page says so, else null",
      "note": "what they do in this piece, one short phrase"}}
  ],
  "places": [
    {{"name": "place name as written",
      "kind": "city|state|country|region|neighborhood|venue|other",
      "note": "what happens there according to the page, one short phrase"}}
  ],
  "organizations": [
    {{"name": "organization name",
      "kind": "government|party|university|school|company|media|ngo|religious|union|cultural|other"}}
  ],
  "events": [
    {{"what": "what happened, one sentence",
      "when": "YYYY-MM-DD, YYYY-MM, YYYY, or the page's own wording",
      "where": "place name or null"}}
  ],
  "facts": ["a specific, checkable assertion the page makes: figures, dates, decisions, names of things"],
  "frames": ["how the piece construes its subject, 1-4 short noun phrases"],
  "quotes": [{{"speaker": "who is speaking", "text": "their words, quoted from the page"}}],
  "published_estimate": "YYYY-MM-DD if the page states or strongly implies its own date, else null",
  "period_start": "earliest date the piece discusses, YYYY-MM-DD / YYYY-MM / YYYY, else null",
  "period_end": "latest date the piece discusses, same formats, else null",
  "temporal_mode": "current_event|retrospective|forecast|timeless",
  "summary": "3-5 sentences: what this piece is about and what it argues or reports"
}}

Rules:
- "role" matters. "author" wrote the piece (a byline). "quoted" is given direct
  speech or is explicitly cited as a source. "subject" is what the piece is
  about. "mentioned" appears in passing. Choose the strongest that applies.
- "quotes" must be speech actually printed on the page. Never invent or
  paraphrase one. An empty list is a correct answer.
- "facts" must be checkable against the page. Not opinions, not your inferences.
- "published_estimate" is when the piece went out. "period_start"/"period_end"
  bound the time the piece *talks about*, which is often earlier. Leave any of
  them null rather than guessing from the capture date above.
- The values for "genre", "intent", "secondary_intents", "stance", "role",
  "kind" and "temporal_mode" must come from the lists above, in English,
  spelled exactly as shown.
- Every other field -- names, places, quotes, facts, frames, summary -- must be
  in the language of the page itself.
- If the page is a navigation index, an error page, or has no readable article,
  return "genre": "listing" with empty lists and say so in "summary".
- Use null or an empty list for anything the page does not support.
- Keep the lists to what matters: at most 12 people, 10 places, 8 organizations,
  8 events, 10 facts, 6 frames and 8 quotes. Choose the most important ones."""


# ===================== validation =====================

def _dicts(value: Any, keys: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Coerce a model-supplied list into records with a fixed shape.

    ``keys`` maps field name to either a permitted-value set (the field is
    treated as an enum and coerced) or None (free text). A record with no
    usable value in its first key is dropped: an entry with no name is noise.
    """
    if not isinstance(value, (list, tuple)):
        return []
    primary = next(iter(keys))
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in value:
        if isinstance(item, str):
            item = {primary: item}
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}
        for field_name, allowed in keys.items():
            raw = item.get(field_name)
            if isinstance(allowed, set):
                record[field_name] = _pick_enum(raw, allowed, None)
            else:
                cleaned = str(raw).strip() if raw not in (None, "") else ""
                record[field_name] = cleaned[:300] or None
        name = record.get(primary)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
        if len(out) >= limit:
            break
    return out


def _date_like(value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    text_value = str(value).strip()
    if text_value.lower() in ("null", "none", "unknown", "n/a"):
        return None
    return text_value[:50]


def validate_reading(data: Any) -> Optional[Dict[str, Any]]:
    """Normalize a close-reading response, or None if it is unusable.

    None routes the page to ERROR so it can be retried, rather than storing an
    empty reading that would silently shrink every aggregate built from it.
    """
    if not isinstance(data, dict):
        return None
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return None

    people = _dicts(data.get("people"),
                    {"name": None, "role": PERSON_ROLES, "affiliation": None, "note": None}, 40)
    for person in people:
        person["role"] = person["role"] or "mentioned"
    places = _dicts(data.get("places"), {"name": None, "kind": PLACE_KINDS, "note": None}, 30)
    for place in places:
        place["kind"] = place["kind"] or "other"
    orgs = _dicts(data.get("organizations"), {"name": None, "kind": ORG_KINDS}, 30)
    for org in orgs:
        org["kind"] = org["kind"] or "other"

    intents = [i for i in _as_str_list(data.get("secondary_intents"), 6) if i.lower() in INTENTS]

    return {
        "title": (str(data.get("title") or "").strip() or None),
        "genre": _pick_enum(data.get("genre"), GENRES, "other"),
        "intent": _pick_enum(data.get("intent"), INTENTS, "other"),
        "secondary_intents": [i.lower() for i in intents],
        "stance": _pick_enum(data.get("stance"), STANCES, "informative"),
        "people": people,
        "places": places,
        "organizations": orgs,
        "events": _dicts(data.get("events"), {"what": None, "when": None, "where": None}, 20),
        "facts": _as_str_list(data.get("facts"), 20),
        "frames": _as_str_list(data.get("frames"), 8),
        "quotes": _dicts(data.get("quotes"), {"speaker": None, "text": None}, 15),
        "published_estimate": _date_like(data.get("published_estimate")),
        "period_start": _date_like(data.get("period_start")),
        "period_end": _date_like(data.get("period_end")),
        "temporal_mode": _pick_enum(data.get("temporal_mode"), TEMPORAL_MODES, None),
        "summary": summary[:5000],
    }


# ===================== the service =====================

@dataclass
class _Page:
    file_id: int
    url: str
    timestamp: Optional[str]
    local_path: str


class ReadingService:
    def __init__(self):
        self._jobs: Dict[int, JobState] = {}

    def job(self, project_id: int) -> JobState:
        return self._jobs.get(project_id) or JobState(kind="read")

    def stop(self, project_id: int) -> Dict[str, Any]:
        state = self._jobs.get(project_id)
        if not state or not state.running:
            return {"status": "not_running"}
        state.stop.set()
        return {"status": "stopping"}

    def start_read(self, project_id: int, limit: int = 0) -> Dict[str, Any]:
        current = self._jobs.get(project_id)
        if current and current.running:
            return {"status": "already_running"}
        state = JobState(kind="read", running=True, started_at=datetime.utcnow())
        self._jobs[project_id] = state

        async def runner():
            try:
                await self.read(project_id, state, limit)
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

    # ----- 1. choose the documents -----
    async def select_pages(self, project_id: int) -> Dict[str, int]:
        """Queue one capture per article-shaped URL.

        The representative is the capture that the triage pass analyzed
        directly (file_analyses.source_file_id IS NULL), so a reading and its
        analysis always describe the same bytes.
        """
        async with async_session_maker() as session:
            rows = (await session.execute(text("""
                SELECT af.id, af.original_url
                FROM archived_files af
                JOIN file_analyses fa ON fa.file_id = af.id
                WHERE af.project_id = :pid AND af.in_scope = 1
                  AND af.status = 'PROCESSED' AND af.local_path IS NOT NULL
                  AND fa.source_file_id IS NULL
            """), {"pid": project_id})).fetchall()

            existing = set((await session.execute(text(
                "SELECT file_id FROM page_readings WHERE project_id = :pid"),
                {"pid": project_id})).scalars().all())

            queued = 0
            for file_id, url in rows:
                if file_id in existing:
                    continue
                link = normalize_link(url, url)
                if not link or classify_link(link, True, 0.0) != "article":
                    continue
                session.add(PageReading(project_id=project_id, file_id=file_id, status="pending"))
                queued += 1
            await session.commit()
        return {"queued": queued, "already_queued": len(existing)}

    # ----- 2. read them -----
    async def read(self, project_id: int, state: Optional[JobState] = None,
                   limit: int = 0) -> Dict[str, Any]:
        state = state or JobState(kind="read")
        cfg = get_llm_config()
        max_calls = limit or cfg.max_calls_per_run or 0

        async with async_session_maker() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise ValueError("Project not found")
            model = project.ollama_model or cfg.model
            state.total = (await session.execute(text(
                "SELECT COUNT(*) FROM page_readings WHERE project_id = :pid AND status = 'pending'"
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
                                         f"endpoint's allowance, limit {ceiling * 100:.0f}%). "
                                         f"Progress is saved.")
                        break
                    state.usage = usage.get("max")

                pages = await self._claim(project_id, room)
                if not pages:
                    break
                state.current = pages[0].url
                results = await asyncio.gather(*[
                    self._read_one(client, cfg, model, page) for page in pages
                ])
                fatal = await self._store(project_id, model, pages, results, state)
                if fatal:
                    state.outcome = "failed"
                    state.message = fatal
                    break

        copied = await self.copy_to_duplicates(project_id)
        if copied:
            state.message = (state.message or "") + f" Copied to {copied} duplicate captures."
        return {"status": state.outcome or "completed", "calls": state.calls}

    async def _claim(self, project_id: int, room: int) -> List[_Page]:
        """Take the next batch, marking it in progress so a restart cannot double-read."""
        async with async_session_maker() as session:
            rows = (await session.execute(text("""
                SELECT pr.file_id, af.original_url, af.timestamp, af.local_path
                FROM page_readings pr JOIN archived_files af ON af.id = pr.file_id
                WHERE pr.project_id = :pid AND pr.status = 'pending'
                ORDER BY af.timestamp, pr.file_id LIMIT :room
            """), {"pid": project_id, "room": room})).fetchall()
            if not rows:
                return []
            pages = [_Page(file_id=r[0], url=r[1], timestamp=r[2], local_path=r[3]) for r in rows]
            await session.execute(
                update(PageReading)
                .where(PageReading.file_id.in_([p.file_id for p in pages]))
                .values(status="reading")
            )
            await session.commit()
        return pages

    async def _read_one(self, client: OllamaClient, cfg: LLMConfig, model: str,
                        page: _Page) -> Tuple[str, Optional[Dict], str, Optional[str], bool, int]:
        """Returns (outcome, data, raw, error, fatal, tokens). Touches no ORM object."""
        try:
            content = await asyncio.to_thread(
                llm_service.read_and_clean, page.local_path, READING_MAX_CHARS)
        except FileNotFoundError:
            return "skipped", None, "", "Capture file is missing from disk", False, 0
        except Exception as exc:
            return "error", None, "", f"{type(exc).__name__}: {exc}", False, 0
        if len(content.strip()) < 200:
            return "skipped", None, "", "Too little text to read", False, 0

        prompt = build_reading_prompt(content, page.url, page.timestamp)
        try:
            result = await client.generate(
                prompt, model=model,
                options={"num_predict": READING_NUM_PREDICT, "temperature": 0.2, "top_p": 0.9},
                fmt="json",
            )
        except OllamaError as exc:
            return "error", None, "", cfg.redact(str(exc)), exc.fatal, 0
        except Exception as exc:
            return "error", None, "", cfg.redact(f"{type(exc).__name__}: {exc}"), False, 0

        raw = result.get("response", "") or ""
        tokens = result.get("eval_count", 0) or 0
        data = validate_reading(llm_service._extract_json(raw))
        if data is None:
            return ("error", None, failure_raw(result),
                    unparseable_error(result, READING_NUM_PREDICT), False, tokens)
        return "processed", data, raw, None, False, tokens

    async def _store(self, project_id: int, model: str, pages: List[_Page],
                     results: List[Tuple], state: JobState) -> Optional[str]:
        fatal = None
        async with async_session_maker() as session:
            rows = (await session.execute(
                select(PageReading).where(PageReading.file_id.in_([p.file_id for p in pages]))
            )).scalars().all()
            by_file = {row.file_id: row for row in rows}
            for page, (outcome, data, raw, error, is_fatal, tokens) in zip(pages, results):
                row = by_file.get(page.file_id)
                if row is None:
                    continue
                state.calls += 1
                state.done += 1
                row.attempts = (row.attempts or 0) + 1
                row.model_used = model
                row.tokens_used = tokens
                row.created_at = datetime.utcnow()
                if outcome == "processed":
                    row.status = "processed"
                    row.error_message = None
                    row.raw_response = None   # only kept for failures; 758 of these is a lot of text
                    for key, value in data.items():
                        setattr(row, key, value)
                elif outcome == "skipped":
                    row.status = "skipped"
                    row.error_message = error
                else:
                    row.status = "error"
                    row.error_message = (error or "")[:500]
                    log.warning("Close reading failed for file %s: %s", row.file_id, error,
                                extra={"project_id": row.project_id})
                    row.raw_response = raw[:4000] if raw else None
                    state.errors += 1
                    if is_fatal:
                        fatal = error
            await session.commit()
        return fatal

    # ----- 3. give the duplicate captures the same reading -----
    async def copy_to_duplicates(self, project_id: int) -> int:
        """Mirror a reading onto other captures of the same URL.

        Same rule as FileAnalysis: the reading describes a URL's content, and
        re-reading the 2014 and 2015 copies of one article would pay twice for
        the same answer. Copies carry ``source_file_id``, and aggregates filter
        them out.
        """
        async with async_session_maker() as session:
            result = await session.execute(text("""
                INSERT INTO page_readings (
                    project_id, file_id, source_file_id, status, title, genre, intent,
                    secondary_intents, stance, people, places, organizations, events,
                    facts, frames, quotes, published_estimate, period_start, period_end,
                    temporal_mode, summary, model_used, created_at)
                SELECT :pid, dup.id, src.file_id, 'processed', src.title, src.genre, src.intent,
                       src.secondary_intents, src.stance, src.people, src.places,
                       src.organizations, src.events, src.facts, src.frames, src.quotes,
                       src.published_estimate, src.period_start, src.period_end,
                       src.temporal_mode, src.summary, src.model_used, CURRENT_TIMESTAMP
                FROM archived_files dup
                JOIN archived_files orig ON orig.original_url = dup.original_url
                     AND orig.project_id = dup.project_id
                JOIN page_readings src ON src.file_id = orig.id
                WHERE dup.project_id = :pid AND dup.in_scope = 1 AND dup.id != orig.id
                  AND src.status = 'processed' AND src.source_file_id IS NULL
                  AND NOT EXISTS (SELECT 1 FROM page_readings x WHERE x.file_id = dup.id)
                GROUP BY dup.id
            """), {"pid": project_id})
            await session.commit()
            return result.rowcount or 0

    async def requeue(self, project_id: int, which: str) -> int:
        """``errors`` retries failures; ``all`` re-reads everything from scratch."""
        async with async_session_maker() as session:
            if which == "errors":
                stmt = (update(PageReading).where(PageReading.status.in_(("error", "reading")))
                        .values(status="pending"))
            elif which == "all":
                # Drop the copies first, or they would survive as stale readings
                # pointing at originals that are about to change.
                await session.execute(text(
                    "DELETE FROM page_readings WHERE project_id = :pid AND source_file_id IS NOT NULL"),
                    {"pid": project_id})
                stmt = update(PageReading).values(status="pending")
            else:
                raise ValueError(f"Unknown queue: {which}")
            result = await session.execute(stmt.where(PageReading.project_id == project_id))
            await session.commit()
            return result.rowcount or 0

    async def stats(self, project_id: int) -> Dict[str, Any]:
        async with async_session_maker() as session:
            status_rows = (await session.execute(text(
                "SELECT status, COUNT(*) FROM page_readings WHERE project_id = :pid "
                "AND source_file_id IS NULL GROUP BY 1"), {"pid": project_id})).fetchall()
            copies = (await session.execute(text(
                "SELECT COUNT(*) FROM page_readings WHERE project_id = :pid "
                "AND source_file_id IS NOT NULL"), {"pid": project_id})).scalar() or 0
        counts = {status: int(n) for status, n in status_rows}
        done = counts.get("processed", 0) + counts.get("error", 0) + counts.get("skipped", 0)
        pending = counts.get("pending", 0) + counts.get("reading", 0)
        return {
            "counts": counts, "copies": copies,
            "total": done + pending, "done": done, "pending": pending,
            "pct": int(100 * done / (done + pending)) if (done + pending) else 0,
        }


reading_service = ReadingService()
