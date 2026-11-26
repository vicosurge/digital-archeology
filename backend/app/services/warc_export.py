"""Export the held captures as a WARC file (ISO 28500), the web archive format.

The zip export is for reading. A WARC is for other archives: pywb, OpenWayback,
ReplayWeb.page, Archives Unleashed and institutional repositories read it
directly.

Records:

- one ``warcinfo`` record describing the file and how it was made;
- one ``resource`` record per held capture: site pages, recovered linked
  pages and recovered images. WARC-Target-URI is the original URL, WARC-Date
  the moment the archive captured it, and WARC-Source-URI the archive address
  the bytes were fetched from;
- one ``metadata`` record per analyzed page, holding the AI analysis as JSON
  and tied to its capture with WARC-Concurrent-To.

They are ``resource`` records, not ``response`` records, on purpose: captures
were fetched in the archive's raw (``id_``) mode, which returns the original
bytes but not the original HTTP headers. A ``response`` record would have to
invent those headers. The warcinfo record says so.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import text

from ..config import resolve_local_path
from .archive_export import EXPORTS_DIR, _slug

SOFTWARE = "Digital Archeology (https://github.com/vicosurge/digital-archeology)"


def warc_date(ts: Optional[str]) -> Optional[str]:
    """14-digit Wayback timestamp -> WARC-Date (ISO 8601, UTC)."""
    digits = "".join(c for c in (ts or "") if c.isdigit())
    if len(digits) < 8:
        return None
    d = digits.ljust(14, "0")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[8:10]}:{d[10:12]}:{d[12:14]}Z"


def raw_url(ts: str, url: str) -> str:
    return f"https://web.archive.org/web/{ts}id_/{url}"


async def _rows(db, project_id: int, all_captures: bool) -> Dict[str, List[Dict[str, Any]]]:
    scope = "" if all_captures else "AND af.in_scope = 1"
    site = (await db.execute(text(f"""
        SELECT af.id, af.original_url, af.timestamp, af.mimetype, af.local_path, af.in_scope,
               fa.id IS NOT NULL, fa.content_type, fa.interaction_type, fa.sentiment,
               fa.sentiment_score, fa.subjects, fa.tags, fa.people_mentioned, fa.description,
               fa.estimated_date, fa.date_confidence, fa.model_used, fa.source_file_id
        FROM archived_files af LEFT JOIN file_analyses fa ON fa.file_id = af.id
        WHERE af.project_id = :pid AND af.local_path IS NOT NULL {scope}
        ORDER BY af.timestamp, af.id
    """), {"pid": project_id})).fetchall()
    links = (await db.execute(text("""
        SELECT url, wayback_timestamp, wayback_url, mimetype, local_path FROM link_targets
        WHERE project_id = :pid AND recovery_status = 'recovered' AND local_path IS NOT NULL
        ORDER BY wayback_timestamp
    """), {"pid": project_id})).fetchall()
    images = (await db.execute(text("""
        SELECT src_url, wayback_timestamp, wayback_url, mimetype, local_path FROM page_images
        WHERE project_id = :pid AND recovery_status = 'recovered' AND local_path IS NOT NULL
        ORDER BY wayback_timestamp
    """), {"pid": project_id})).fetchall()

    def js(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except ValueError:
                return v
        return v

    out_site = []
    for r in site:
        analysis = None
        if r[6]:
            analysis = {"content_type": r[7], "interaction_type": r[8], "sentiment": r[9],
                        "sentiment_score": r[10], "subjects": js(r[11]), "tags": js(r[12]),
                        "people_mentioned": js(r[13]), "description": r[14],
                        "estimated_date": r[15], "date_confidence": r[16], "model": r[17],
                        "shared_with_capture": r[18]}
        out_site.append({"url": r[1], "ts": r[2], "mimetype": r[3], "path": r[4],
                         "source": raw_url(r[2], r[1]), "in_scope": bool(r[5]), "analysis": analysis,
                         "kind": "site"})
    extra = [{"url": u, "ts": ts, "source": wb or (raw_url(ts, u) if ts else None), "mimetype": m,
              "path": p, "kind": kind, "analysis": None}
             for kind, rows in (("link", links), ("image", images)) for u, ts, wb, m, p in rows]
    return {"site": out_site, "extra": extra}


def write_warc(project: Dict[str, Any], rows: Dict[str, List[Dict[str, Any]]], out_path: str) -> Dict[str, int]:
    """Write the WARC. Synchronous: run it in a thread."""
    from warcio.warcwriter import WARCWriter

    counts = {"resource": 0, "metadata": 0, "missing": 0, "duplicate": 0}
    seen = set()
    tmp = out_path + ".part"
    with open(tmp, "wb") as fh:
        writer = WARCWriter(fh, gzip=True, warc_version="1.1")
        info = {
            "software": SOFTWARE,
            "format": "WARC File Format 1.1",
            "description": (f"Captures of {project['domain']} held by a Digital Archeology project, "
                            "fetched from public web archives (the Internet Archive's Wayback Machine "
                            "unless WARC-Source-URI says otherwise)."),
            "conformsTo": "https://iipc.github.io/warc-specifications/specifications/warc-format/warc-1.1/",
            "record-types": ("resource records hold the archived bytes as the archive returned them in "
                             "raw (id_) mode; the original HTTP headers were not available, so no "
                             "response records are written. metadata records hold AI-generated analyses "
                             "(finding aids, not evidence), linked by WARC-Concurrent-To."),
            "scope": project.get("scope_note") or "all held captures",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        writer.write_record(writer.create_warcinfo_record(os.path.basename(out_path), info))
        for item in rows["site"] + rows["extra"]:
            date = warc_date(item["ts"])
            ident = (item["url"], item["ts"])
            if not item["url"] or not date:
                counts["missing"] += 1
                continue
            if ident in seen:
                counts["duplicate"] += 1
                continue
            try:
                path = resolve_local_path(item["path"])
                size = os.path.getsize(path)
            except (OSError, TypeError):
                counts["missing"] += 1
                continue
            seen.add(ident)
            headers = {"WARC-Date": date}
            if item.get("source"):
                headers["WARC-Source-URI"] = item["source"]
            with open(path, "rb") as payload:
                record = writer.create_warc_record(
                    item["url"], "resource", payload=payload, length=size,
                    warc_content_type=item.get("mimetype") or "application/octet-stream",
                    warc_headers_dict=headers)
                writer.write_record(record)
            counts["resource"] += 1
            if item.get("analysis"):
                body = json.dumps({"kind": item["kind"], "in_scope": item.get("in_scope"),
                                   **item["analysis"]}, ensure_ascii=False).encode("utf-8")
                meta = writer.create_warc_record(
                    item["url"], "metadata", payload=io.BytesIO(body), length=len(body),
                    warc_content_type="application/json",
                    warc_headers_dict={"WARC-Date": date,
                                       "WARC-Concurrent-To": record.rec_headers.get_header("WARC-Record-ID")})
                writer.write_record(meta)
                counts["metadata"] += 1
    os.replace(tmp, out_path)
    return counts


async def build_warc(db, project, all_captures: bool = False) -> Dict[str, Any]:
    rows = await _rows(db, project.id, all_captures)
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(EXPORTS_DIR, f"{_slug(project.domain)}-{stamp}.warc.gz")
    note = None
    if not all_captures and (project.scope_start or project.scope_end):
        note = (f"in-scope captures only ({project.scope_start or 'start'} to {project.scope_end or 'end'}): "
                f"{project.scope_reason or ''}").strip()
    counts = await asyncio.to_thread(write_warc, {"domain": project.domain, "scope_note": note}, rows, path)
    return {"path": path, **counts}


def latest_warc(project) -> Optional[str]:
    if not os.path.isdir(EXPORTS_DIR):
        return None
    prefix = f"{_slug(project.domain)}-"
    found = sorted(f for f in os.listdir(EXPORTS_DIR)
                   if f.startswith(prefix) and f.endswith(".warc.gz") and "-archive-" not in f)
    return os.path.join(EXPORTS_DIR, found[-1]) if found else None
