"""How one page changed across its captures.

Every held capture of a URL (all the spellings the archive used for it), oldest
first, each compared with the one before on its visible text: unchanged, or
how many words were added and removed. Any two captures can be compared side
by side. Text only: markup changes (a new ad slot, a theme update) are not
what a historian asks about, and they would flag every capture as changed.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from . import local_view
from .llm import llm_service

MAX_CHARS = 200_000
MAX_CAPTURES = 150


def _read(path: str) -> str:
    try:
        return llm_service.read_and_clean(path, MAX_CHARS)
    except OSError:
        return ""


def word_change(old: str, new: str) -> Dict[str, int]:
    """Words added and removed between two texts."""
    a, b = old.split(), new.split()
    added = removed = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            removed += i2 - i1
        if op in ("replace", "insert"):
            added += j2 - j1
    return {"added": added, "removed": removed}


def sentences(value: str) -> List[str]:
    """Split text into short lines so a side-by-side diff lines up."""
    parts = re.split(r"(?<=[.!?…])\s+|\s{2,}|\s(?=[•·|])", value)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        while len(p) > 160:
            cut = p.rfind(" ", 0, 160)
            cut = cut if cut > 40 else 160
            out.append(p[:cut])
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def side_by_side(old: str, new: str, old_label: str, new_label: str) -> str:
    """An HTML table of the differences (difflib escapes the text itself)."""
    return difflib.HtmlDiff(wrapcolumn=70).make_table(
        sentences(old), sentences(new), old_label, new_label, context=True, numlines=2)


async def captures(db, project_id: int, url: str) -> List[Dict[str, Any]]:
    variants = local_view.url_variants(url)
    marks = ",".join(f":v{i}" for i in range(len(variants)))
    rows = (await db.execute(text(f"""
        SELECT id, original_url, timestamp, local_path, mimetype, in_scope
        FROM archived_files
        WHERE project_id = :pid AND local_path IS NOT NULL AND original_url IN ({marks})
        ORDER BY timestamp
    """), {"pid": project_id, **{f"v{i}": v for i, v in enumerate(variants)}})).fetchall()
    return [dict(id=r[0], url=r[1], timestamp=r[2], local_path=r[3], mimetype=r[4],
                 in_scope=bool(r[5])) for r in rows]


async def capture_count(db, project_id: int, url: str) -> int:
    variants = local_view.url_variants(url)
    marks = ",".join(f":v{i}" for i in range(len(variants)))
    return (await db.execute(text(f"""
        SELECT COUNT(*) FROM archived_files
        WHERE project_id = :pid AND local_path IS NOT NULL AND original_url IN ({marks})
    """), {"pid": project_id, **{f"v{i}": v for i, v in enumerate(variants)}})).scalar() or 0


async def timeline(db, project_id: int, url: str) -> Dict[str, Any]:
    caps = await captures(db, project_id, url)
    truncated = len(caps) > MAX_CAPTURES
    if truncated:
        # Keep the first and the most recent ones; the middle is what thins out.
        caps = caps[:10] + caps[-(MAX_CAPTURES - 10):]
    texts = await asyncio.gather(*(asyncio.to_thread(_read, c["local_path"]) for c in caps))
    prev: Optional[str] = None
    versions = 0
    for cap, body in zip(caps, texts):
        cap["words"] = len(body.split())
        cap["digest"] = hashlib.sha1(body.encode("utf-8", "ignore")).hexdigest()[:10]
        if prev is None:
            cap["change"] = "first"
            versions += 1
        elif body == prev:
            cap["change"] = "same"
        else:
            cap.update(change="changed", **await asyncio.to_thread(word_change, prev, body))
            versions += 1
        prev = body
    return {"captures": caps, "versions": versions, "truncated": truncated}


async def compare(db, project_id: int, a_id: int, b_id: int) -> Optional[Dict[str, Any]]:
    rows = (await db.execute(text(
        "SELECT id, timestamp, local_path FROM archived_files "
        "WHERE project_id = :pid AND id IN (:a, :b) AND local_path IS NOT NULL"),
        {"pid": project_id, "a": a_id, "b": b_id})).fetchall()
    by_id = {r[0]: r for r in rows}
    if a_id not in by_id or b_id not in by_id:
        return None
    a, b = by_id[a_id], by_id[b_id]
    old, new = await asyncio.gather(asyncio.to_thread(_read, a[2]), asyncio.to_thread(_read, b[2]))

    def label(ts: str) -> str:
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}"

    if old == new:
        table = ""
    else:
        table = await asyncio.to_thread(side_by_side, old, new, label(a[1]), label(b[1]))
    return {"a": {"id": a[0], "timestamp": a[1]}, "b": {"id": b[0], "timestamp": b[1]},
            "table": table, **(word_change(old, new) if old != new else {"added": 0, "removed": 0})}
