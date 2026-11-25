"""How the studied site changed over its lifetime, year by year.

Everything here is aggregated from analyses that already exist, so it costs no
LLM calls. Years are *capture* years: the Wayback timestamp says when a page
was online, which is reliable, while the LLM's estimated publication date is
often a guess. Only in-scope, directly-analyzed rows count (copies made for
duplicate captures would inflate every figure, see models.FileAnalysis).
"""
from collections import Counter
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_BASE = """
    FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id
    WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND af.in_scope = 1
      AND fa.source_file_id IS NULL
"""

_PERSON_VALUE = (
    "CASE WHEN json_valid(je.value) AND json_type(je.value) = 'object' "
    "THEN json_extract(je.value, '$.name') ELSE je.value END"
)


async def _top_json_values(db: AsyncSession, project_id: int, column: str,
                           per_year: int, value_sql: str = "je.value") -> Dict[str, List[tuple]]:
    rows = (await db.execute(text(f"""
        SELECT substr(af.timestamp, 1, 4) AS yr, {value_sql} AS v, COUNT(*) AS n
        FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id,
             json_each(fa.{column}) je
        WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND af.in_scope = 1
          AND fa.source_file_id IS NULL AND v IS NOT NULL AND trim(v) != ''
        GROUP BY yr, lower(trim(v))
        ORDER BY yr, n DESC
    """), {"pid": project_id})).fetchall()
    out: Dict[str, List[tuple]] = {}
    for yr, value, n in rows:
        bucket = out.setdefault(yr, [])
        if len(bucket) < per_year:
            bucket.append((value, int(n)))
    return out


async def compute_themes(db: AsyncSession, project_id: int) -> Dict[str, Any]:
    params = {"pid": project_id}
    year_rows = (await db.execute(text(f"""
        SELECT substr(af.timestamp, 1, 4) AS yr,
               COUNT(*) AS pages,
               AVG(fa.sentiment_score) AS avg_score,
               SUM(fa.sentiment = 'positive') AS pos,
               SUM(fa.sentiment = 'negative') AS neg,
               SUM(fa.sentiment = 'neutral') AS neu,
               SUM(fa.sentiment = 'mixed') AS mixed
        {_BASE} AND af.timestamp IS NOT NULL
        GROUP BY yr ORDER BY yr
    """), params)).fetchall()

    content_rows = (await db.execute(text(f"""
        SELECT substr(af.timestamp, 1, 4) AS yr, fa.content_type, COUNT(*)
        {_BASE} AND fa.content_type IS NOT NULL AND fa.content_type != ''
        GROUP BY yr, fa.content_type ORDER BY yr, COUNT(*) DESC
    """), params)).fetchall()
    content_by_year: Dict[str, List[tuple]] = {}
    for yr, ctype, n in content_rows:
        content_by_year.setdefault(yr, []).append((ctype, int(n)))

    subjects = await _top_json_values(db, project_id, "subjects", 10)
    people = await _top_json_values(db, project_id, "people_mentioned", 10, _PERSON_VALUE)

    # Which outside sites the site pointed to each year (from recovered links'
    # citations; a citation is dated by the citing capture).
    cited_rows = (await db.execute(text("""
        SELECT substr(af.timestamp, 1, 4) AS yr, lt.domain, COUNT(DISTINCT af.original_url) AS pages
        FROM page_links pl
        JOIN link_targets lt ON lt.id = pl.target_id
        JOIN archived_files af ON af.id = pl.source_file_id
        WHERE pl.project_id = :pid AND lt.is_internal = 0
          AND lt.kind IN ('reference', 'document')
        GROUP BY yr, lt.domain ORDER BY yr, pages DESC
    """), params)).fetchall()
    cited_by_year: Dict[str, List[tuple]] = {}
    for yr, domain, n in cited_rows:
        bucket = cited_by_year.setdefault(yr, [])
        if len(bucket) < 8:
            bucket.append((domain, int(n)))

    # People who appear in a year but in no earlier year: arrivals, which is
    # often more telling than the raw top list.
    seen: set = set()
    years: List[Dict[str, Any]] = []
    for yr, pages, avg_score, pos, neg, neu, mixed in year_rows:
        names = people.get(yr, [])
        new_people = [n for n, _ in names if n.lower() not in seen]
        seen.update(n.lower() for n, _ in names)
        years.append({
            "year": yr,
            "pages": int(pages),
            "avg_sentiment": round(avg_score, 2) if avg_score is not None else None,
            "sentiment": {"positive": int(pos or 0), "negative": int(neg or 0),
                          "neutral": int(neu or 0), "mixed": int(mixed or 0)},
            "content_types": content_by_year.get(yr, [])[:5],
            "subjects": subjects.get(yr, []),
            "people": names,
            "new_people": new_people,
            "cited_domains": cited_by_year.get(yr, []),
        })

    return {"years": years, "total_pages": sum(y["pages"] for y in years)}


def sentiment_shift(years: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The largest year-over-year change in average sentiment, for the headline."""
    best = None
    for prev, cur in zip(years, years[1:]):
        if prev["avg_sentiment"] is None or cur["avg_sentiment"] is None:
            continue
        # Years with a handful of pages swing wildly; ignore them.
        if min(prev["pages"], cur["pages"]) < 30:
            continue
        delta = cur["avg_sentiment"] - prev["avg_sentiment"]
        if best is None or abs(delta) > abs(best["delta"]):
            best = {"from": prev["year"], "to": cur["year"], "delta": round(delta, 2)}
    return best or {}
