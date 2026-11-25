"""Serve downloaded captures locally, so the site can be read without the Wayback Machine.

A page's links and images are rewritten to point back into this viewer. The
viewer resolves each URL to the closest-in-time capture held on disk (site
pages, recovered page images, recovered linked material). What is not held
locally falls back to the Wayback Machine.

Archived pages are third-party HTML. They are served with a CSP sandbox, so
their scripts never run on the app's origin, and without a same-origin grant.
"""
from __future__ import annotations

import html
import os
import re
from typing import List, Optional, Tuple
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from sqlalchemy import text

from ..config import resolve_local_path

SANDBOX_HEADERS = {
    "Content-Security-Policy": "sandbox allow-popups allow-popups-to-escape-sandbox",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

_ATTR_RE = re.compile(
    rb"""(?P<pre>\s(?:href|src|background|action|lowsrc)\s*=\s*)"""
    rb"""(?P<val>"[^"]*"|'[^']*'|[^\s>"']+)""",
    re.IGNORECASE,
)
_TAG_RE = re.compile(rb"<[a-zA-Z][^>]*>")
_BASE_RE = re.compile(rb"""<base\s[^>]*href\s*=\s*["']?([^"'\s>]+)""", re.IGNORECASE)
_BASE_TAG_RE = re.compile(rb"<base\b[^>]*>", re.IGNORECASE)
_BODY_RE = re.compile(rb"<body[^>]*>", re.IGNORECASE)
_SKIP_SCHEMES = ("javascript:", "mailto:", "data:", "tel:", "about:", "#")


def url_variants(url: str) -> List[str]:
    """Spellings the CDX index may have used for the same address."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if not host:
        return [url]
    bare = host[4:] if host.startswith("www.") else host
    hosts = {bare, "www." + bare, bare + ":80", "www." + bare + ":80"}
    path = parts.path or "/"
    paths = {path}
    if path.endswith("/"):
        paths |= {path + "index.html", path + "index.htm", path.rstrip("/") or "/"}
    else:
        paths.add(path + "/")
        for idx in ("index.html", "index.htm", "default.htm"):
            if path.lower().endswith("/" + idx):
                paths.add(path[: -len(idx)])
    out = []
    for scheme in ("http", "https"):
        for h in hosts:
            for p in paths:
                out.append(urlunsplit((scheme, h, p, parts.query, "")))
    return out


def local_href(project_id: int, ts: str, url: str) -> str:
    return f"/project/{project_id}/local?ts={ts}&url={quote(url, safe='')}"


async def resolve(db, project_id: int, url: str, ts: str) -> Optional[Tuple[str, str, str, str]]:
    """Find the held capture of ``url`` closest to ``ts``.

    Returns (path_on_disk, mimetype, capture_timestamp, kind) or None.
    """
    variants = url_variants(url)
    marks = ",".join(f":v{i}" for i in range(len(variants)))
    params = {f"v{i}": v for i, v in enumerate(variants)}
    params.update(pid=project_id, ts=int(ts) if ts.isdigit() else 0)
    queries = (
        ("page", f"""SELECT local_path, mimetype, timestamp FROM archived_files
                    WHERE project_id=:pid AND local_path IS NOT NULL AND original_url IN ({marks})
                    ORDER BY ABS(CAST(timestamp AS INTEGER) - :ts) LIMIT 1"""),
        ("image", f"""SELECT local_path, mimetype, wayback_timestamp FROM page_images
                     WHERE project_id=:pid AND local_path IS NOT NULL AND src_url IN ({marks})
                     ORDER BY ABS(CAST(wayback_timestamp AS INTEGER) - :ts) LIMIT 1"""),
        ("linked", f"""SELECT local_path, mimetype, wayback_timestamp FROM link_targets
                      WHERE project_id=:pid AND local_path IS NOT NULL AND url IN ({marks})
                      LIMIT 1"""),
    )
    for kind, sql in queries:
        row = (await db.execute(text(sql), params)).first()
        if row:
            path = resolve_local_path(row[0])
            if path and os.path.isfile(path):
                return path, row[1] or "", row[2] or ts, kind
    return None


def rewrite_html(body: bytes, page_url: str, ts: str, project_id: int) -> bytes:
    """Point every link and image of an archived page back into the viewer.

    Works on bytes so the page's own encoding (often windows-1252 in old
    sites) survives untouched; URLs are ASCII in practice.
    """
    base = page_url
    m = _BASE_RE.search(body)
    if m:
        base = urljoin(page_url, html.unescape(m.group(1).decode("latin-1")))
    body = _BASE_TAG_RE.sub(b"", body)

    def attr(mm: re.Match) -> bytes:
        raw = mm.group("val")
        val = raw[1:-1] if raw[:1] in (b'"', b"'") else raw
        s = html.unescape(val.decode("latin-1")).strip()
        if not s or s.lower().startswith(_SKIP_SCHEMES):
            return mm.group(0)
        target, _, frag = urljoin(base, s).partition("#")
        if not target.lower().startswith(("http://", "https://")):
            return mm.group(0)
        new = local_href(project_id, ts, target) + (f"#{frag}" if frag else "")
        return mm.group("pre") + b'"' + html.escape(new).encode("ascii", "ignore") + b'"'

    return _TAG_RE.sub(lambda t: _ATTR_RE.sub(attr, t.group(0)), body)


def banner(project_id: int, url: str, capture_ts: str, kind: str) -> bytes:
    date = f"{capture_ts[:4]}-{capture_ts[4:6]}-{capture_ts[6:8]}" if len(capture_ts) >= 8 else capture_ts
    wb = f"https://web.archive.org/web/{capture_ts}/{url}"
    label = {"page": "Local copy", "linked": "Local copy of linked page"}.get(kind, "Local copy")
    e = html.escape
    return (
        '<div style="all:initial;display:block;font:13px/1.5 system-ui,sans-serif;'
        'background:#1f2937;color:#f9fafb;padding:6px 12px;border-bottom:2px solid #f59e0b">'
        f'<b>{label}</b> · captured {e(date)} · <span style="opacity:.8">{e(url)}</span> · '
        f'<a style="color:#fcd34d" href="{e(wb)}" target="_blank">Wayback Machine</a> · '
        f'<a style="color:#fcd34d" href="/project/{project_id}">Back to project</a></div>'
    ).encode("utf-8")


def with_banner(body: bytes, bar: bytes) -> bytes:
    m = _BODY_RE.search(body)
    if m:
        return body[: m.end()] + bar + body[m.end():]
    return bar + body


def html_charset(body: bytes) -> Optional[str]:
    """utf-8 when the bytes are valid utf-8 and the page does not say otherwise."""
    if re.search(rb"<meta[^>]+charset", body[:4096], re.IGNORECASE):
        return None
    try:
        body.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "windows-1252"


def not_held_page(project_id: int, url: str, ts: str) -> str:
    e = html.escape
    wb = f"https://web.archive.org/web/{ts}/{url}"
    return (
        '<!doctype html><meta charset="utf-8"><title>Not held locally</title>'
        '<body style="font:15px/1.6 system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px">'
        f'<h2>Not held locally</h2><p><code>{e(url)}</code> was not downloaded into this project.</p>'
        f'<p><a href="{e(wb)}">Look for it on the Wayback Machine</a> · '
        f'<a href="/project/{project_id}">Back to project</a></p>'
    )
