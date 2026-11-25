"""Images on pages that have no text of their own.

Analysis skips a page with under MIN_CONTENT_CHARS of text as "too little
content". For an early site that is often a photo page -- Revista Espiral's
2002 galleries are a bare <img> each -- so the skip is right for the text
model but the page's content is the picture. Three steps, each resumable:

    select    record every <img> on the skipped pages
    recover   fetch each image from the Wayback Machine (same replay endpoint
              and pace as link recovery); an image the archive never kept is
              recorded as not_archived, which is itself a finding
    describe  send each recovered image to a vision model (gemma4 on
              ollama.com reports the capability) for a plain description

The same image URL is fetched and described once per project and the result
copied to every page that shows it.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

import aiofiles
import httpx
from sqlalchemy import select, text, update

from ..config import get_llm_config, resolve_local_path
from ..database import DATABASE_DIR, async_session_maker
from ..models import ArchivedFile, FileStatus, PageImage, Project
from .links import (RECOVER_REQUESTS_PER_MINUTE, JobState, TransientWaybackError,
                    _file_extension, _sleep_unless, _timestamp_from_wayback_url,
                    wayback_view_url)
from .llm import failure_raw, llm_service, unparseable_error
from .llm_client import OllamaClient, OllamaError, RateLimiter

import logging as _logging
log = _logging.getLogger("app.images")

IMAGES_DIR = os.path.join(DATABASE_DIR, "images")
# Spacers, bullets and buttons: not worth a vision call.
MIN_IMAGE_BYTES = 2048
# Formats the vision endpoint accepts. GIFs (common on 2002 sites) are kept
# but not described.
DESCRIBABLE = ("image/jpeg", "image/png", "image/webp")
DESCRIBE_NUM_PREDICT = 700
SKIPPED_REASON = "Too little content"


class _ImgParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images: List[Tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "img":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        if a.get("src"):
            self.images.append((a["src"].strip(), a.get("alt", "").strip()))


def extract_images(path: str, page_url: str) -> List[Tuple[str, str]]:
    """(absolute src, alt) for every <img> on the page, de-duplicated."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        parser = _ImgParser()
        parser.feed(fh.read())
    seen, out = set(), []
    for src, alt in parser.images:
        if src.startswith(("data:", "mmres:", "javascript:")):
            continue
        absolute = urljoin(page_url, src)
        if urlsplit(absolute).scheme not in ("http", "https") or absolute in seen:
            continue
        seen.add(absolute)
        out.append((absolute, alt))
    return out


def build_image_prompt(page_url: str, page_title: Optional[str], alt: Optional[str],
                       captured: Optional[str]) -> str:
    return f"""You are helping a historian describe one image from an archived website.
Describe only what can be seen. Do not guess who people are unless their name
is written in the image itself.

Page the image appeared on: {page_url}
Page title: {page_title or "none"}
Alt text in the page: {alt or "none"}
Date the page was captured by the web archive: {captured or "unknown"}

Answer with this JSON object and nothing else:

{{
  "description": "two to four sentences: what the image shows (people, setting, objects, artwork, layout), in the language of the page title if it has one",
  "visible_text": "any text legible in the image, transcribed exactly, or an empty string",
  "kind": "photograph|artwork|illustration|scan_of_print|logo|banner|screenshot|other"
}}"""


def _fmt_ts(ts: Optional[str]) -> Optional[str]:
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}" if ts and len(ts) >= 8 else None


def _page_title(path: Optional[str]) -> Optional[str]:
    path = resolve_local_path(path)
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        m = re.search(r"<title[^>]*>(.*?)</title>", fh.read(20000), re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:200] if m else None


class ImageService:
    # ----- select -----
    async def select(self, project_id: int) -> Dict[str, Any]:
        """Record the images on every in-scope page skipped for lack of text."""
        async with async_session_maker() as session:
            pages = (await session.execute(
                select(ArchivedFile.id, ArchivedFile.local_path, ArchivedFile.original_url)
                .where(ArchivedFile.project_id == project_id)
                .where(ArchivedFile.status == FileStatus.SKIPPED)
                .where(ArchivedFile.error_message == SKIPPED_REASON)
                .where(ArchivedFile.in_scope.is_(True))
            )).all()
            existing = {(r[0], r[1]) for r in (await session.execute(
                select(PageImage.file_id, PageImage.src_url)
                .where(PageImage.project_id == project_id))).all()}
            added = pages_with_images = 0
            for file_id, local_path, url in pages:
                path = resolve_local_path(local_path)
                if not path or not os.path.exists(path):
                    continue
                found = await asyncio.to_thread(extract_images, path, url)
                if found:
                    pages_with_images += 1
                for src, alt in found:
                    if (file_id, src) in existing:
                        continue
                    session.add(PageImage(project_id=project_id, file_id=file_id,
                                          src_url=src, alt_text=alt or None))
                    added += 1
            await session.commit()
        return {"pages_checked": len(pages), "pages_with_images": pages_with_images,
                "images_added": added}

    # ----- recover -----
    async def recover(self, project_id: int, state: Optional[JobState] = None,
                      limit: int = 0) -> Dict[str, Any]:
        state = state or JobState(kind="img-recover")
        out_dir = os.path.join(IMAGES_DIR, str(project_id))
        os.makedirs(out_dir, exist_ok=True)
        async with async_session_maker() as session:
            # One fetch per distinct URL; the earliest page that shows it dates the request.
            rows = (await session.execute(text("""
                SELECT pi.src_url, MIN(af.timestamp)
                FROM page_images pi JOIN archived_files af ON af.id = pi.file_id
                WHERE pi.project_id = :pid AND pi.recovery_status = 'pending'
                GROUP BY pi.src_url ORDER BY pi.src_url
            """), {"pid": project_id})).all()
        if limit:
            rows = rows[:limit]
        state.total = len(rows)
        pace = RateLimiter(RECOVER_REQUESTS_PER_MINUTE)
        headers = {"User-Agent": "DigitalArcheology/1.0 (historical research; image recovery)"}
        backoff = 30.0
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0),
                                     follow_redirects=True, headers=headers) as client:
            for index, (src, page_ts) in enumerate(rows):
                if state.stop.is_set():
                    break
                state.current = src
                while True:
                    try:
                        values = await self._recover_one(client, pace, src, page_ts,
                                                         out_dir, project_id, index)
                        backoff = 30.0
                        break
                    except TransientWaybackError as exc:
                        state.message = f"Wayback busy ({exc}); waiting {int(backoff)}s"
                        await _sleep_unless(state.stop, backoff)
                        backoff = min(backoff * 2, 300.0)
                        state.message = None
                        if state.stop.is_set():
                            values = None
                            break
                if values is None:
                    break
                async with async_session_maker() as session:
                    await session.execute(
                        update(PageImage)
                        .where(PageImage.project_id == project_id)
                        .where(PageImage.src_url == src)
                        .values(**values, attempts=PageImage.attempts + 1)
                    )
                    await session.commit()
                state.done += 1
                if values["recovery_status"] == "error":
                    state.errors += 1
                    log.warning("Image recovery failed: %s", values.get("error_message"),
                                extra={"project_id": project_id})
        return await self.stats(project_id)

    async def _recover_one(self, client: httpx.AsyncClient, pace: RateLimiter, src: str,
                           page_ts: Optional[str], out_dir: str, project_id: int,
                           index: int) -> Dict[str, Any]:
        await pace.acquire()
        ts = (page_ts or "")[:14] or "2000"
        try:
            response = await client.get(f"https://web.archive.org/web/{ts}id_/{src}")
        except httpx.TimeoutException:
            return {"recovery_status": "error", "error_message": "Timed out fetching the image"}
        except httpx.TransportError as exc:
            raise TransientWaybackError(f"{type(exc).__name__} fetching image") from exc
        if response.status_code in (429, 502, 503, 504):
            raise TransientWaybackError(f"HTTP {response.status_code}")
        if response.status_code in (403, 404):
            return {"recovery_status": "not_archived", "error_message": None}
        if response.status_code >= 400:
            return {"recovery_status": "error",
                    "error_message": f"HTTP {response.status_code} fetching image"}
        mimetype = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not mimetype.startswith("image/"):
            # The archive answered with a page (its own error page, or the site's).
            return {"recovery_status": "not_archived",
                    "error_message": f"Archive returned {mimetype or 'no type'}, not an image"}
        content = response.content
        final_ts = _timestamp_from_wayback_url(str(response.url)) or ts
        common = {"mimetype": mimetype, "file_size": len(content), "wayback_timestamp": final_ts,
                  "wayback_url": wayback_view_url(final_ts, src)}
        if len(content) < MIN_IMAGE_BYTES:
            return {**common, "recovery_status": "skipped",
                    "error_message": f"Decorative ({len(content)} bytes)"}
        path = os.path.join(out_dir, f"{index:05d}{_file_extension(mimetype, src)}")
        async with aiofiles.open(path, "wb") as fh:
            await fh.write(content)
        status = "pending" if mimetype in DESCRIBABLE else "skipped"
        return {**common, "recovery_status": "recovered", "local_path": path,
                "error_message": None, "description_status": status}

    # ----- describe -----
    async def describe(self, project_id: int, state: Optional[JobState] = None,
                       limit: int = 0) -> Dict[str, Any]:
        state = state or JobState(kind="img-describe")
        cfg = get_llm_config()
        async with async_session_maker() as session:
            project = await session.get(Project, project_id)
            model = project.ollama_model or cfg.model
            rows = (await session.execute(text("""
                SELECT pi.src_url, MIN(pi.local_path), MIN(pi.alt_text),
                       MIN(af.original_url), MIN(af.local_path), MIN(af.timestamp)
                FROM page_images pi JOIN archived_files af ON af.id = pi.file_id
                WHERE pi.project_id = :pid AND pi.description_status = 'pending'
                GROUP BY pi.src_url ORDER BY pi.src_url
            """), {"pid": project_id})).all()
        if limit:
            rows = rows[:limit]
        state.total = len(rows)
        async with OllamaClient(cfg, RateLimiter(cfg.requests_per_minute)) as client:
            for src, img_path, alt, page_url, page_path, page_ts in rows:
                if state.stop.is_set():
                    break
                state.current = src
                values = await self._describe_one(client, cfg, model, img_path, alt,
                                                  page_url, page_path, page_ts)
                state.calls += 1
                async with async_session_maker() as session:
                    await session.execute(
                        update(PageImage)
                        .where(PageImage.project_id == project_id)
                        .where(PageImage.src_url == src)
                        .values(**values, model_used=model, described_at=datetime.utcnow())
                    )
                    await session.commit()
                state.done += 1
                if values["description_status"] == "error":
                    state.errors += 1
                    log.warning("Image description failed: %s", values.get("error_message"),
                                extra={"project_id": project_id})
                    if values.pop("_fatal", False):
                        state.outcome = "failed"
                        state.message = values.get("error_message")
                        break
        return await self.stats(project_id)

    async def _describe_one(self, client: OllamaClient, cfg, model: str, img_path: str,
                            alt: Optional[str], page_url: str, page_path: Optional[str],
                            page_ts: Optional[str]) -> Dict[str, Any]:
        path = resolve_local_path(img_path)
        if not path or not os.path.exists(path):
            return {"description_status": "error", "error_message": "Image missing from disk"}
        async with aiofiles.open(path, "rb") as fh:
            b64 = base64.b64encode(await fh.read()).decode("ascii")
        prompt = build_image_prompt(page_url, _page_title(page_path), alt, _fmt_ts(page_ts))
        try:
            result = await client.generate(
                prompt, model=model, fmt="json", images=[b64],
                options={"num_predict": DESCRIBE_NUM_PREDICT, "temperature": 0.2},
            )
        except OllamaError as exc:
            return {"description_status": "error", "error_message": cfg.redact(str(exc))[:500],
                    "_fatal": exc.fatal}
        tokens = result.get("eval_count", 0) or 0
        data = llm_service._extract_json(result.get("response") or "")
        if not isinstance(data, dict) or not str(data.get("description") or "").strip():
            return {"description_status": "error", "tokens_used": tokens,
                    "error_message": unparseable_error(result, DESCRIBE_NUM_PREDICT),
                    "raw_response": failure_raw(result)[:4000]}
        return {
            "description_status": "processed", "tokens_used": tokens, "error_message": None,
            "description": str(data["description"]).strip()[:3000],
            "visible_text": (str(data.get("visible_text") or "").strip()[:2000]) or None,
            "raw_response": json.dumps(data, ensure_ascii=False)[:4000],
        }

    # ----- reporting -----
    async def stats(self, project_id: int) -> Dict[str, Any]:
        async with async_session_maker() as session:
            rows = (await session.execute(text("""
                SELECT recovery_status, description_status,
                       COUNT(*), COUNT(DISTINCT src_url), COUNT(DISTINCT file_id)
                FROM page_images WHERE project_id = :pid GROUP BY 1, 2 ORDER BY 1, 2
            """), {"pid": project_id})).all()
        return {"groups": [{"recovery": r[0], "description": r[1], "rows": r[2],
                            "distinct_images": r[3], "pages": r[4]} for r in rows]}

    async def requeue(self, project_id: int, which: str) -> int:
        """``errors``: recovery errors. ``describe_errors``: failed descriptions."""
        async with async_session_maker() as session:
            if which == "errors":
                stmt = (update(PageImage).where(PageImage.recovery_status == "error")
                        .values(recovery_status="pending"))
            elif which == "describe_errors":
                stmt = (update(PageImage).where(PageImage.description_status == "error")
                        .values(description_status="pending"))
            else:
                raise ValueError(f"Unknown queue: {which}")
            result = await session.execute(stmt.where(PageImage.project_id == project_id))
            await session.commit()
            return result.rowcount or 0


image_service = ImageService()
