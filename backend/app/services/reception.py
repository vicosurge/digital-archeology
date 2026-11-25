"""Reception: what readers did with the site, recovered from the markup.

Every other pass here studies what a publication said. This one studies what
came back. Reader comments are the only trace of an audience a static archive
preserves, and they are usually thrown away -- stripped as boilerplate along
with the navigation, or worse, folded into the article text so that a reader
arguing in the comments is counted as a person the piece is "about".

Two decisions shape this module.

**No LLM.** WordPress comment markup is regular: an ``ol.commentlist`` of
``li#li-comment-N`` items, each with an author, a relative timestamp and a
body, and a ``bypostauthor`` class when the author is the site itself. Parsing
it is exact, free, costs no quota and can be audited by anyone who opens the
capture. A model asked to "find the comments" would produce an estimate where
the markup gives a fact.

**The last capture wins.** An article's text is fixed once published, so the
close reading takes whichever capture the triage pass analyzed. Comments are
not fixed: they accrete for as long as the page is live, and an early capture
shows an empty thread under a piece that later drew a dozen replies. Comments
are therefore read from the *latest* in-scope capture of each URL, and
deduplicated on the comment id WordPress assigns, so a thread captured five
times is one thread.

The timestamps are relative ("8 days ago") because the theme renders them that
way. Relative to the capture date they become absolute, which is a date the
Wayback record does not otherwise give up: not when the page was archived, but
when a reader wrote something on it.
"""
import asyncio
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ..config import resolve_local_path
from ..database import async_session_maker
from ..models import PageComment, Project
from .links import JobState

# "8 days ago", "1 day ago", "3 months ago", and the Spanish rendering some
# themes emit. Anything else is kept verbatim and simply left undated.
_AGO_RE = re.compile(
    r"(?:hace\s+)?(\d+)\s+(second|minute|hour|day|week|month|year|"
    r"segundo|minuto|hora|d[ií]a|semana|mes|a[nñ]o)s?(?:\s+ago)?",
    re.IGNORECASE)

_UNIT_DAYS = {
    "second": 0, "segundo": 0, "minute": 0, "minuto": 0, "hour": 0, "hora": 0,
    "day": 1, "dia": 1, "día": 1, "week": 7, "semana": 7,
    "month": 30, "mes": 30, "year": 365, "ano": 365, "año": 365,
}


def page_key(url: str) -> str:
    """Fold the URL spellings of one page together.

    One article is captured under several addresses: with and without a
    trailing slash, and with WordPress's own ``?replytocom=`` reply links,
    which are separate URLs pointing at the same page and the same thread.
    Treating them as distinct pages counted 103 comments where there were 75.
    """
    key = re.sub(r"^https?://", "", url or "", flags=re.IGNORECASE)
    key = re.sub(r"^www\.", "", key, flags=re.IGNORECASE)
    key = key.replace(":80/", "/").split("?")[0].split("#")[0]
    return key.rstrip("/").lower()


def _fold(value: str) -> str:
    """Accent- and case-insensitive key, matching apparatus.fold."""
    stripped = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(stripped.lower().replace(".", " ").split())


# Themes append this to the author name ("Rodrigo said…").
_SAID_RE = re.compile(r"\s*(said|dijo|comentó|escribió)\s*[.…:]*\s*$", re.IGNORECASE)


def parse_relative_time(label: str, captured: Optional[str]) -> Optional[str]:
    """Turn 'N days ago' plus a capture timestamp into a YYYY-MM-DD date.

    Approximate by construction: months are 30 days and anything under a day
    resolves to the capture date. That is honest for what it is -- the page
    never stated a real date, and a reader wanting precision has the capture
    and the comment id to work from.
    """
    if not label or not captured or len(captured) < 8:
        return None
    match = _AGO_RE.search(label)
    if not match:
        return None
    try:
        base = datetime.strptime(captured[:8], "%Y%m%d")
    except ValueError:
        return None
    unit = match.group(2).lower()
    days = _UNIT_DAYS.get(unit)
    if days is None:
        return None
    return (base - timedelta(days=int(match.group(1)) * days)).strftime("%Y-%m-%d")


@dataclass
class ParsedComment:
    ref: str                     # the id WordPress gave it, stable across captures
    author: str = ""
    body: str = ""
    time_label: str = ""
    depth: int = 1
    by_site: bool = False        # the publication answering in its own comments
    author_url: str = ""


class _CommentParser(HTMLParser):
    """Pull the comment thread out of a WordPress page.

    Scoped to ``ol/ul.commentlist`` on purpose. Sidebars carry a "recent
    comments" widget repeating the same remarks site-wide; counting those would
    inflate every figure here exactly as sitewide banners inflated the link
    counts before they were classified out.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.comments: List[ParsedComment] = []
        self._in_list = 0            # nesting depth inside commentlist
        self._list_tag: List[str] = []
        self._stack: List[ParsedComment] = []
        self._section: Optional[str] = None
        self._section_depth = 0
        self._buf: List[str] = []
        self._skip = 0               # inside script/style

    # -- helpers --
    @staticmethod
    def _classes(attrs) -> List[str]:
        for key, value in attrs:
            if key == "class" and value:
                return value.split()
        return []

    @staticmethod
    def _attr(attrs, name) -> str:
        for key, value in attrs:
            if key == name and value:
                return value
        return ""

    def _flush(self) -> str:
        chunk = " ".join(" ".join(self._buf).split())
        self._buf = []
        return chunk

    # -- parsing --
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
            return
        classes = self._classes(attrs)

        if not self._in_list:
            if tag in ("ol", "ul") and "commentlist" in classes:
                self._in_list = 1
                self._list_tag = [tag]
            return

        if tag in ("ol", "ul"):
            self._in_list += 1
            self._list_tag.append(tag)
            return

        if tag == "li" and any(c == "comment" or c.startswith("comment-") for c in classes):
            ident = self._attr(attrs, "id") or ""
            ref = ident.replace("li-comment-", "").replace("comment-", "") or str(len(self.comments) + 1)
            depth = 1
            for cls in classes:
                if cls.startswith("depth-") and cls[6:].isdigit():
                    depth = int(cls[6:])
            self._stack.append(ParsedComment(
                ref=ref, depth=depth, by_site="bypostauthor" in classes))
            return

        if not self._stack:
            return

        if self._section:
            # Track nesting so a </div> inside the body does not end the section.
            if tag == self._section_tag:
                self._section_depth += 1
            return

        for name in ("comment-author", "comment-meta", "comment-body", "comment-content"):
            if name in classes:
                self._section = name
                self._section_tag = tag
                self._section_depth = 1
                self._buf = []
                return

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
            return
        if not self._in_list:
            return

        if self._section and tag == self._section_tag:
            self._section_depth -= 1
            if self._section_depth <= 0:
                value = self._flush()
                current = self._stack[-1] if self._stack else None
                if current:
                    if self._section == "comment-author" and not current.author:
                        current.author = _SAID_RE.sub("", value).strip()
                    elif self._section == "comment-meta" and not current.time_label:
                        current.time_label = value
                    elif self._section in ("comment-body", "comment-content") and not current.body:
                        current.body = value
                self._section = None
            return

        if tag == "li" and self._stack:
            done = self._stack.pop()
            if done.author or done.body:
                self.comments.append(done)
            return

        if tag in ("ol", "ul") and self._list_tag:
            self._list_tag.pop()
            self._in_list -= 1
            if self._in_list <= 0:
                self._in_list = 0
                self._stack = []

    def handle_data(self, data):
        if self._skip or not self._section:
            return
        self._buf.append(data)


# The "Reply" and relative-date links live inside comment-meta; strip the noise
# so what remains is the timestamp the theme printed.
_META_NOISE_RE = re.compile(r"\b(reply|responder|permalink|enlace|edit|editar)\b\s*", re.IGNORECASE)


def parse_comments(html: str) -> List[ParsedComment]:
    parser = _CommentParser()
    try:
        parser.feed(html)
    except Exception:
        # A malformed capture should cost this page's comments, not the run.
        pass
    out = []
    for comment in parser.comments:
        comment.author = comment.author.strip()[:200]
        comment.body = comment.body.strip()[:4000]
        comment.time_label = _META_NOISE_RE.sub("", comment.time_label).replace("·", "").strip()[:100]
        if comment.author or comment.body:
            out.append(comment)
    return out


class ReceptionService:
    def __init__(self):
        self._jobs: Dict[int, JobState] = {}

    def job(self, project_id: int) -> JobState:
        return self._jobs.get(project_id) or JobState(kind="reception")

    async def extract(self, project_id: int, state: Optional[JobState] = None) -> Dict[str, Any]:
        """Read the comment threads off the latest capture of every in-scope URL."""
        state = state or JobState(kind="reception")
        async with async_session_maker() as session:
            # The theme marks the publication's own replies with `bypostauthor`,
            # but not consistently -- b1mas4 answered readers 17 times and only
            # 12 carry the class. The account name is the reliable tell, so the
            # site's own name is matched as well. Getting this wrong would count
            # the editors among their own audience.
            project = await session.get(Project, project_id)
            site_names = set()
            if project:
                site_names.add(_fold(project.domain.split("/")[0].split(":")[0]))
                site_names.add(_fold(project.domain.split(".")[0]))
                site_names.add(_fold(project.name))
            # One capture per URL: the last one, because comments accrete.
            candidates = (await session.execute(text("""
                SELECT af.id, af.original_url, af.timestamp, af.local_path
                FROM archived_files af
                WHERE af.project_id = :pid AND af.in_scope = 1 AND af.local_path IS NOT NULL
                ORDER BY af.timestamp
            """), {"pid": project_id})).fetchall()
            # One capture per *page*, not per URL spelling, and the latest one,
            # because comments accrete. Ordering by timestamp above means the
            # last write for a key wins.
            latest: Dict[str, tuple] = {}
            for row in candidates:
                latest[page_key(row[1])] = row
            rows = list(latest.values())
            state.total = len(rows)

            await session.execute(text("DELETE FROM page_comments WHERE project_id = :pid"),
                                  {"pid": project_id})
            await session.commit()

            found = 0
            pending: List[PageComment] = []
            for file_id, url, timestamp, local_path in rows:
                if state.stop.is_set():
                    break
                state.done += 1
                state.current = url
                comments = await asyncio.to_thread(self._read_file, local_path)
                for comment in comments:
                    pending.append(PageComment(
                        project_id=project_id, file_id=file_id, page_url=url,
                        page_key=page_key(url),
                        capture_timestamp=timestamp, comment_ref=comment.ref,
                        author=comment.author or None, body=comment.body or None,
                        time_label=comment.time_label or None,
                        commented_on=parse_relative_time(comment.time_label, timestamp),
                        depth=comment.depth,
                        by_site=comment.by_site or _fold(comment.author) in site_names,
                    ))
                    found += 1
                if len(pending) >= 200:
                    session.add_all(pending)
                    await session.commit()
                    pending = []
            if pending:
                session.add_all(pending)
            await session.commit()
        return {"captures_scanned": state.done, "comments": found}

    @staticmethod
    def _read_file(local_path: str) -> List[ParsedComment]:
        path = resolve_local_path(local_path)
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                html = fh.read()
        except OSError:
            return []
        if "commentlist" not in html:
            return []
        return parse_comments(html)


reception_service = ReceptionService()
