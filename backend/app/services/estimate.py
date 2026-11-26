"""What a run is about to cost, shown before it starts.

Calls are exact: they are the rows each job would send to the model. Tokens are
an estimate. Output comes from this project's (or, failing that, any project's)
past replies for the same model; input comes from past ``prompt_tokens`` where
they were recorded, otherwise from file sizes: page text is roughly a third of
the HTML, capped at what is actually sent, at about four characters a token,
plus the prompt's own instructions.

Dollars appear only when LLM_PRICE_INPUT_PER_MTOK / LLM_PRICE_OUTPUT_PER_MTOK
are set. Local Ollama costs nothing, and a made-up price would be worse than
none.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_llm_config, get_prices

CHARS_PER_TOKEN = 4.0
TEXT_SHARE_OF_HTML = 0.35
DEFAULT_OUTPUT_TOKENS = 250
# Instructions and JSON schema sent with every call, in tokens.
PROMPT_OVERHEAD = {"analysis": 450, "links": 550, "readings": 900, "images": 250}
# A vision model's cost for one image, in input tokens.
IMAGE_INPUT_TOKENS = 300
READING_MAX_CHARS = 12000

JOBS = ("analysis", "links", "readings", "images")


def _input_from_size(avg_bytes: Optional[float], max_chars: int, job: str) -> float:
    text_chars = min((avg_bytes or 0) * TEXT_SHARE_OF_HTML, max_chars)
    return text_chars / CHARS_PER_TOKEN + PROMPT_OVERHEAD[job]


def summarize(calls: int, in_per_call: float, out_per_call: float,
              prices: tuple, basis: str) -> Dict[str, Any]:
    in_tokens = int(round(calls * in_per_call))
    out_tokens = int(round(calls * out_per_call))
    price_in, price_out = prices
    dollars = None
    if price_in or price_out:
        dollars = round(in_tokens / 1e6 * price_in + out_tokens / 1e6 * price_out, 2)
    return {"calls": calls, "in_tokens": in_tokens, "out_tokens": out_tokens,
            "dollars": dollars, "basis": basis}


async def _avg(db: AsyncSession, sql: str, params: Dict[str, Any]) -> Optional[float]:
    value = (await db.execute(text(sql), params)).scalar()
    return float(value) if value else None


async def _output_avg(db: AsyncSession, table: str, project_id: int, model: Optional[str],
                      model_col: str = "model_used") -> tuple:
    """(average output tokens, where the figure came from)."""
    base = f"SELECT AVG(tokens_used) FROM {table} WHERE tokens_used > 0"
    if model:
        v = await _avg(db, f"{base} AND project_id = :pid AND {model_col} = :m",
                       {"pid": project_id, "m": model})
        if v:
            return v, "this project's past replies"
        v = await _avg(db, f"{base} AND {model_col} = :m", {"m": model})
        if v:
            return v, f"past replies from {model}"
    return DEFAULT_OUTPUT_TOKENS, "a default reply size"


async def _analysis_output(db: AsyncSession, project_id: int, model: Optional[str]) -> tuple:
    # file_analyses has no project_id: join through archived_files.
    sql = ("SELECT AVG(fa.tokens_used) FROM file_analyses fa "
           "JOIN archived_files af ON af.id = fa.file_id WHERE fa.tokens_used > 0")
    if model:
        v = await _avg(db, sql + " AND af.project_id = :pid AND fa.model_used = :m",
                       {"pid": project_id, "m": model})
        if v:
            return v, "this project's past replies"
        v = await _avg(db, sql + " AND fa.model_used = :m", {"m": model})
        if v:
            return v, f"past replies from {model}"
    return DEFAULT_OUTPUT_TOKENS, "a default reply size"


async def estimate(db: AsyncSession, project_id: int, job: str,
                   model: Optional[str] = None) -> Dict[str, Any]:
    cfg = get_llm_config()
    prices = get_prices()
    p = {"pid": project_id}

    if job == "analysis":
        from .processing import _analyzable_sql_list
        row = (await db.execute(text(f"""
            SELECT COUNT(DISTINCT original_url), AVG(file_size) FROM archived_files
            WHERE project_id = :pid AND status = 'DOWNLOADED' AND in_scope = 1
              AND local_path IS NOT NULL
              AND (mimetype IN ({_analyzable_sql_list()}) OR mimetype IS NULL)
        """), p)).fetchone()
        calls, avg_size = int(row[0] or 0), row[1]
        out, basis = await _analysis_output(db, project_id, model)
        measured_in = await _avg(db, "SELECT AVG(fa.prompt_tokens) FROM file_analyses fa "
                                     "JOIN archived_files af ON af.id = fa.file_id "
                                     "WHERE af.project_id = :pid AND fa.prompt_tokens > 0", p)
        in_per = measured_in or _input_from_size(avg_size, cfg.max_content_chars, job)
        return summarize(calls, in_per, out, prices, basis)

    if job == "links":
        row = (await db.execute(text(
            "SELECT COUNT(*), AVG(file_size) FROM link_targets "
            "WHERE project_id = :pid AND analysis_status = 'pending'"), p)).fetchone()
        # link_targets keeps no token counts; its replies are the same size of
        # JSON as a page analysis, so that average stands in.
        out, basis = await _analysis_output(db, project_id, model)
        return summarize(int(row[0] or 0), _input_from_size(row[1], cfg.max_content_chars, job),
                         out, prices, basis)

    if job == "readings":
        row = (await db.execute(text(
            "SELECT COUNT(*), AVG(af.file_size) FROM page_readings pr "
            "JOIN archived_files af ON af.id = pr.file_id "
            "WHERE pr.project_id = :pid AND pr.status = 'pending'"), p)).fetchone()
        out, basis = await _output_avg(db, "page_readings", project_id, model)
        return summarize(int(row[0] or 0), _input_from_size(row[1], READING_MAX_CHARS, job),
                         out, prices, basis)

    if job == "images":
        calls = (await db.execute(text(
            "SELECT COUNT(*) FROM page_images "
            "WHERE project_id = :pid AND description_status = 'pending'"), p)).scalar() or 0
        out, basis = await _output_avg(db, "page_images", project_id, model)
        return summarize(int(calls), IMAGE_INPUT_TOKENS + PROMPT_OVERHEAD[job], out, prices, basis)

    raise ValueError(f"unknown job {job!r}")


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def describe(e: Dict[str, Any]) -> str:
    """One line for the UI and the CLI."""
    if not e["calls"]:
        return ""
    line = (f"About {e['calls']:,} calls · ~{fmt_tokens(e['in_tokens'])} in / "
            f"{fmt_tokens(e['out_tokens'])} out tokens")
    if e["dollars"] is not None:
        line += f" · ≈ ${e['dollars']:,.2f}"
    return line + f" (estimate, from {e['basis']})"
