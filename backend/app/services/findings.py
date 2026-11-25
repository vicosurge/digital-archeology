"""Evidence-backed findings about the studied site.

The archival card's summary is a paragraph a reader has to take on trust. A
historian can't use that. Here the model gets numbered evidence -- summaries of
the site's own pages (P<id>) and of what it linked to (L<id>) -- and every
claim it returns must cite those numbers. Citations that don't match anything
it was shown are dropped, and a claim left with no valid citation is marked
unsupported instead of being quietly presented as fact.

One call to the LLM per build: cheap enough to rebuild after new analyses land.
"""
import re
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_llm_config
from ..models import Project
from .enrichment import build_timeline
from .links import wayback_view_url
from .llm import llm_service
from .llm_client import OllamaClient
from .themes import compute_themes

PAGES_PER_YEAR = 14
MAX_LINKS = 30
ASSESSMENTS = {"achieved", "partially_achieved", "not_achieved", "unclear"}

# Pages that say who a site is and what it wants: always worth showing the model.
SELF_DESCRIPTION_RE = re.compile(
    r"(integrantes|quienes|nosotros|about|acerca|mision|manifiesto|equipo|staff|colaboradores|contacto)",
    re.I,
)
EVIDENCE_RE = re.compile(r"^[PL]\d+$")
INLINE_CITES_RE = re.compile(r"\s*[\[(]((?:[PL]\d+)(?:\s*[,;]\s*[PL]\d+)*)[\])]")


async def _page_evidence(db: AsyncSession, project_id: int) -> List[Dict[str, Any]]:
    rows = (await db.execute(text("""
        SELECT af.id, af.original_url, af.timestamp, fa.content_type, fa.sentiment,
               fa.description, fa.subjects
        FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id
        WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND af.in_scope = 1
          AND fa.source_file_id IS NULL AND fa.description IS NOT NULL
        ORDER BY af.timestamp
    """), {"pid": project_id})).fetchall()

    by_year: Dict[str, List[tuple]] = {}
    self_pages: List[tuple] = []
    for row in rows:
        year = (row[2] or "????")[:4]
        if SELF_DESCRIPTION_RE.search(row[1] or ""):
            self_pages.append(row)
        by_year.setdefault(year, []).append(row)

    chosen: Dict[int, tuple] = {r[0]: r for r in self_pages[:10]}
    for year, year_rows in sorted(by_year.items()):
        # Spread picks across the year and across content types rather than
        # taking the first N, which would all come from one month.
        seen_types: Dict[str, int] = {}
        step = max(1, len(year_rows) // (PAGES_PER_YEAR * 2))
        picks = 0
        for row in year_rows[::step]:
            ctype = row[3] or "other"
            if seen_types.get(ctype, 0) >= 4:
                continue
            seen_types[ctype] = seen_types.get(ctype, 0) + 1
            chosen.setdefault(row[0], row)
            picks += 1
            if picks >= PAGES_PER_YEAR:
                break

    evidence = []
    for fid, url, ts, ctype, sentiment, desc, _subjects in sorted(chosen.values(), key=lambda r: r[2] or ""):
        evidence.append({
            "id": f"P{fid}", "type": "page", "url": url, "timestamp": ts,
            "wayback_url": wayback_view_url(ts, url), "content_type": ctype,
            "sentiment": sentiment, "description": (desc or "")[:500],
        })
    return evidence


async def _link_evidence(db: AsyncSession, project_id: int) -> List[Dict[str, Any]]:
    rows = (await db.execute(text("""
        SELECT id, url, domain, kind, cited_by_pages, first_cited_ts, wayback_timestamp,
               title, description, relation, relation_note, is_internal
        FROM link_targets
        WHERE project_id = :pid AND analysis_status = 'processed'
          AND COALESCE(content_type, '') != 'unavailable' AND relation != 'unrelated'
        ORDER BY is_internal, cited_by_pages DESC
        LIMIT :n
    """), {"pid": project_id, "n": MAX_LINKS})).fetchall()
    return [{
        "id": f"L{r[0]}", "type": "link", "url": r[1], "domain": r[2], "kind": r[3],
        "cited_by_pages": r[4], "first_cited_ts": r[5], "timestamp": r[6],
        "wayback_url": wayback_view_url(r[6], r[1]), "title": r[7],
        "description": (r[8] or "")[:400], "relation": r[9], "relation_note": r[10],
        "is_internal": bool(r[11]),
    } for r in rows]


def _fmt_year_line(y: Dict[str, Any]) -> str:
    subjects = ", ".join(v for v, _ in y["subjects"][:8]) or "—"
    people = ", ".join(f"{v} ({n})" for v, n in y["people"][:8]) or "—"
    types = ", ".join(f"{t} {n}" for t, n in y["content_types"]) or "—"
    cited = ", ".join(d for d, _ in y["cited_domains"][:6]) or "—"
    return (f"{y['year']}: {y['pages']} pages analyzed; average sentiment {y['avg_sentiment']}; "
            f"content: {types}; topics: {subjects}; people: {people}; linked outside sites: {cited}")


def build_prompt(project: Project, timeline: Dict[str, Any], themes: Dict[str, Any],
                 pages: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> str:
    page_lines = "\n".join(
        f"[{p['id']}] {p['timestamp'][:8] if p['timestamp'] else '?'} {p['url']} "
        f"({p['content_type']}, {p['sentiment']}): {p['description']}"
        for p in pages
    )
    link_lines = "\n".join(
        f"[{l['id']}] {l['url']} — linked from {l['cited_by_pages']} pages, first "
        f"{(l['first_cited_ts'] or '?')[:8]}; relation: {l['relation']}. "
        f"{l['title'] or ''}: {l['description']} {l['relation_note'] or ''}"
        for l in links
    ) or "(no linked material analyzed yet)"
    years = "\n".join(_fmt_year_line(y) for y in themes["years"])

    return f"""You are assisting a historian who is studying a defunct website from its Internet Archive captures.
Everything you know about the site is below. Do not use outside knowledge about it.

Site: {project.domain}
Period under study: {timeline.get('first_seen')} to {timeline.get('last_seen')}
Why this period: {project.scope_reason or 'not stated'}

Year by year (aggregated from AI analyses of the site's pages):
{years}

Evidence — summaries of the site's own pages:
{page_lines}

Evidence — pages the site linked to, recovered as they looked when cited:
{link_lines}

Write findings for the historian as a JSON object. Every claim MUST cite the evidence ids
(e.g. "P123", "L45") that support it. Cite only ids that appear above. If the evidence does
not support a claim, do not make it; say what is unknown in "open_questions" instead.

{{
  "overview": {{"text": "2-3 sentences: what this website was", "evidence": ["P..."]}},
  "purpose_and_audience": {{"text": "what it set out to do and for whom", "evidence": []}},
  "evolution": [{{"period": "e.g. 2012-2013", "text": "how the site's content, tone or focus changed", "evidence": []}}],
  "people": [{{"name": "person's name", "role": "their apparent role, e.g. founder, editor, contributor, interviewee", "evidence": []}}],
  "network": {{"text": "who and what the site connected with: collaborators, sources, events it promoted", "evidence": []}},
  "goals_and_outcome": {{"assessment": "achieved/partially_achieved/not_achieved/unclear", "text": "did it achieve what it set out to do, based on the evidence", "evidence": []}},
  "decline": {{"text": "what the evidence shows about how and when it wound down", "evidence": []}},
  "open_questions": ["questions the historian should investigate that this evidence cannot answer"]
}}

Write in the same language as most of the site's content. Keep enum values in English.
Respond ONLY with the JSON object."""


def _clean_claim(claim: Any, valid: set) -> Optional[Dict[str, Any]]:
    if not isinstance(claim, dict):
        return None
    text_value = str(claim.get("text") or "").strip()
    raw_ids = claim.get("evidence") or []
    if isinstance(raw_ids, str):
        raw_ids = re.findall(r"[PL]\d+", raw_ids)
    # Models repeat citations inline ("... la región [P12, L4]."). Keep them as
    # evidence, but out of the prose, which renders them as links separately.
    inline = INLINE_CITES_RE.findall(text_value)
    raw_ids = list(raw_ids) + [t for group in inline for t in re.findall(r"[PL]\d+", group)]
    text_value = re.sub(r"\s+([.,;:])", r"\1", INLINE_CITES_RE.sub("", text_value)).strip()
    if not text_value:
        return None
    cited = []
    dropped = 0
    for item in raw_ids:
        token = str(item).strip().strip("[]").upper()
        if EVIDENCE_RE.match(token) and token in valid:
            if token not in cited:
                cited.append(token)
        else:
            dropped += 1
    out = {"text": text_value[:2000], "evidence": cited, "unsupported": not cited}
    if dropped:
        out["invalid_citations"] = dropped
    return out


def validate_findings(data: Any, valid: set) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("overview", "purpose_and_audience", "network", "decline"):
        claim = _clean_claim(data.get(key), valid)
        if claim:
            out[key] = claim

    goals = data.get("goals_and_outcome")
    claim = _clean_claim(goals, valid)
    if claim:
        assessment = str((goals or {}).get("assessment") or "").strip().lower().replace(" ", "_")
        claim["assessment"] = assessment if assessment in ASSESSMENTS else "unclear"
        out["goals_and_outcome"] = claim

    evolution = []
    for item in data.get("evolution") or []:
        claim = _clean_claim(item, valid)
        if claim:
            claim["period"] = str(item.get("period") or "").strip()[:40]
            evolution.append(claim)
    out["evolution"] = evolution

    people = []
    for item in data.get("people") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        claim = _clean_claim({"text": item.get("role") or "role unclear",
                              "evidence": item.get("evidence")}, valid)
        claim["name"] = str(item["name"]).strip()[:200]
        claim["role"] = claim.pop("text")
        people.append(claim)
    out["people"] = people[:25]

    out["open_questions"] = [str(q).strip()[:500] for q in (data.get("open_questions") or [])
                             if str(q).strip()][:12]
    if "overview" not in out:
        return None
    return out


async def build_findings(db: AsyncSession, project: Project) -> Tuple[Dict[str, Any], str]:
    """Run the synthesis. Returns (findings, model). Raises on endpoint failure."""
    cfg = get_llm_config()
    model = project.ollama_model or cfg.model

    timestamps = [r[0] for r in (await db.execute(text(
        "SELECT timestamp FROM archived_files WHERE project_id = :pid AND in_scope = 1 "
        "AND timestamp IS NOT NULL"), {"pid": project.id})).fetchall()]
    timeline = build_timeline(timestamps)
    themes = await compute_themes(db, project.id)
    pages = await _page_evidence(db, project.id)
    links = await _link_evidence(db, project.id)
    if not pages:
        raise ValueError("No analyzed pages in scope yet; run the analysis first.")

    sources = {e["id"]: e for e in pages + links}
    prompt = build_prompt(project, timeline, themes, pages, links)

    call_cfg = replace(cfg, request_timeout=max(cfg.request_timeout, 300.0), max_retries=2)
    findings = None
    raw = ""
    async with OllamaClient(call_cfg) as client:
        for temperature in (0.3, 0.1):
            result = await client.generate(
                prompt, model=model, fmt="json",
                options={"num_predict": 3000, "temperature": temperature, "top_p": 0.9},
            )
            raw = result.get("response", "") or ""
            findings = validate_findings(llm_service._extract_json(raw), set(sources))
            if findings:
                break
    if not findings:
        raise ValueError("The model did not return usable findings. Try rebuilding.")

    cited = set()
    for value in findings.values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                cited.update(item.get("evidence") or [])
    findings["sources"] = {k: v for k, v in sources.items() if k in cited}
    findings["evidence_shown"] = {"pages": len(pages), "links": len(links)}
    findings["generated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return findings, model
