"""The "small archive": a self-contained package a historian can take away.

It opens in any browser with no server and no network, and it is packaged as a
BagIt bag (RFC 8493), the Library of Congress format archives use to move
collections around: every file is listed with its SHA-256, so anyone can verify
nothing was altered after export.

Layout inside the zip::

    <bag>/bagit.txt, bag-info.txt, manifest-sha256.txt
    <bag>/data/index.html          findings, year-by-year themes, how to cite
    <bag>/data/pages.html          every analyzed page of the site, by year
    <bag>/data/links.html          what the site linked to, recovered and classified
    <bag>/data/people.html         people named across the site
    <bag>/data/reading.html        the close readings: who, where, when, what for
    <bag>/data/chronology.html     events the site reported, dated by when they happened
    <bag>/data/methodology.html    what was done, by whom (person or AI), and the limits
    <bag>/data/*.csv, *.json       the same data, machine-readable
    <bag>/data/prompts.txt         the exact wording put to the model
    <bag>/data/cross-check.json    the loss figure tested against other archives
    <bag>/data/captures/site/      raw captures of the site's pages
    <bag>/data/captures/linked/    raw captures of linked material

Raw HTML captures are stored with a .txt suffix. Opened directly, an archived
page would try to load scripts and images from domains that may now belong to
someone else (b1mas4.com itself became a spam farm). The rendered page is one
click away on the Wayback Machine, which is also the citable form.
"""
import asyncio
import csv
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import resolve_local_path
from ..database import DATABASE_DIR
from ..models import ArchivalCard, Project
from . import apparatus
from .enrichment import build_timeline, person_to_str
from .prompt_book import build_prompt_book
from .links import wayback_view_url
from .themes import compute_themes, sentiment_shift

EXPORTS_DIR = os.path.join(DATABASE_DIR, "exports")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "export")


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts or len(ts) < 8 or not ts[:8].isdigit():
        return ts or ""
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"


_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")


def _fmt_month(value: str) -> str:
    """YYYYMM -> 'April 2012'."""
    if not value or len(value) < 6 or not value[:6].isdigit():
        return value or ""
    month = int(value[4:6])
    return f"{_MONTHS[month - 1]} {value[:4]}" if 1 <= month <= 12 else value


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60] or "project"


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR),
                      autoescape=select_autoescape(["html"]))
    env.filters["fmt_ts"] = _fmt_ts
    env.filters["fmt_month"] = _fmt_month
    env.filters["evidence_href"] = lambda token: (
        f"pages.html#{token}" if token.startswith("P") else f"links.html#{token}")
    return env


def _json_any(value: Any) -> List[Any]:
    """Decode a JSON array column, keeping whatever shape its elements have.

    Unlike _json_list this does not flatten objects to names: a reading's
    people carry a role and an affiliation, and losing those is losing the
    point of having read them.
    """
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []


def _json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    if not isinstance(value, list):
        return []
    return [n for n in (person_to_str(v) for v in value) if n]


def _raw_name(prefix: str, ident: int, local_path: str, mimetype: Optional[str]) -> str:
    ext = os.path.splitext(local_path)[1].lower()
    htmlish = (mimetype or "").startswith(("text/html", "application/xhtml")) or ext in (".html", ".htm", "")
    if htmlish:
        return f"captures/{prefix}/{ident}.html.txt"
    return f"captures/{prefix}/{ident}{ext or '.bin'}"


async def gather(db: AsyncSession, project: Project) -> Dict[str, Any]:
    pid = project.id
    card = (await db.execute(select(ArchivalCard).where(ArchivalCard.project_id == pid))).scalar_one_or_none()
    themes = await compute_themes(db, pid)

    timestamps = [r[0] for r in (await db.execute(text(
        "SELECT timestamp FROM archived_files WHERE project_id = :pid AND in_scope = 1 "
        "AND timestamp IS NOT NULL"), {"pid": pid})).fetchall()]
    counts = dict((await db.execute(text(
        "SELECT in_scope, COUNT(*) FROM archived_files WHERE project_id = :pid GROUP BY in_scope"
    ), {"pid": pid})).fetchall())

    page_rows = (await db.execute(text("""
        SELECT af.id, af.original_url, af.timestamp, af.mimetype, af.local_path,
               fa.content_type, fa.interaction_type, fa.sentiment, fa.sentiment_score,
               fa.subjects, fa.tags, fa.people_mentioned, fa.description,
               fa.estimated_date, fa.date_confidence, fa.model_used,
               (SELECT COUNT(*) FROM archived_files s WHERE s.project_id = af.project_id
                  AND s.original_url = af.original_url AND s.in_scope = 1) AS captures
        FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id
        WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND af.in_scope = 1
          AND fa.source_file_id IS NULL
        ORDER BY af.timestamp, af.id
    """), {"pid": pid})).fetchall()

    pages = []
    people_index: Dict[str, Dict[str, Any]] = {}
    from .people import canonicalizer
    canon = await canonicalizer(db, pid)
    for r in page_rows:
        page = {
            "id": f"P{r[0]}", "file_id": r[0], "url": r[1], "timestamp": r[2],
            "date": _fmt_ts(r[2]), "year": (r[2] or "")[:4],
            "wayback_url": wayback_view_url(r[2], r[1]), "mimetype": r[3],
            "local_path": r[4], "content_type": r[5], "interaction_type": r[6],
            "sentiment": r[7], "sentiment_score": r[8],
            "subjects": _json_list(r[9]), "tags": _json_list(r[10]),
            "people": _json_list(r[11]), "description": r[12],
            "estimated_date": r[13], "date_confidence": r[14], "model": r[15],
            "captures": r[16],
        }
        page["raw_file"] = _raw_name("site", r[0], r[4] or "", r[3]) if r[4] else None
        pages.append(page)
        for raw in page["people"]:
            if not person_to_str(raw):
                continue
            k = canon.canonical(raw)
            entry = people_index.setdefault(k, {"name": canon.display(k), "pages": [],
                                                "variants": canon.variants(k)})
            if page["id"] not in entry["pages"]:
                entry["pages"].append(page["id"])

    link_rows = (await db.execute(text("""
        SELECT id, url, domain, is_internal, kind, cited_by_pages, first_cited_ts, last_cited_ts,
               sample_anchor, recovery_status, wayback_timestamp, capture_distance_days, mimetype,
               local_path, analysis_status, title, description, content_type, sentiment,
               subjects, tags, people_mentioned, relation, relation_note, model_used,
               source_archive, wayback_url
        FROM link_targets
        WHERE project_id = :pid
          AND (kind IN ('reference', 'article', 'document', 'sitewide')
               OR recovery_status = 'recovered')
          AND recovery_status NOT IN ('in_corpus', 'not_selected')
        ORDER BY is_internal,
                 CASE kind WHEN 'reference' THEN 0 WHEN 'document' THEN 1 WHEN 'article' THEN 2 ELSE 3 END,
                 cited_by_pages DESC, id
    """), {"pid": pid})).fetchall()

    link_ids = [r[0] for r in link_rows]
    citing: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    analyzed_page_ids = {p["file_id"]: p["id"] for p in pages}
    url_to_page = {p["url"]: p["id"] for p in pages}
    if link_ids:
        rows = (await db.execute(text("""
            SELECT pl.target_id, af.id, af.original_url, af.timestamp, pl.anchor_text
            FROM page_links pl JOIN archived_files af ON af.id = pl.source_file_id
            WHERE pl.project_id = :pid
            ORDER BY pl.target_id, af.timestamp
        """), {"pid": pid})).fetchall()
        wanted = set(link_ids)
        seen = defaultdict(set)
        for tid, fid, url, ts, anchor in rows:
            # The same page is often captured as both /slug and /slug/.
            page_key = re.sub(r"^https?://(www\.)?", "", url or "").replace(":80/", "/").rstrip("/").lower()
            if tid not in wanted or page_key in seen[tid] or len(citing[tid]) >= 25:
                continue
            seen[tid].add(page_key)
            citing[tid].append({
                "page_id": analyzed_page_ids.get(fid) or url_to_page.get(url),
                "url": url, "date": _fmt_ts(ts), "anchor": anchor,
                "wayback_url": wayback_view_url(ts, url),
            })

    links = []
    for r in link_rows:
        recovered = r[9] == "recovered"
        link = {
            "id": f"L{r[0]}", "target_id": r[0], "url": r[1], "domain": r[2],
            "scope": "internal" if r[3] else "external", "kind": r[4],
            "cited_by_pages": r[5], "first_cited": _fmt_ts(r[6]), "last_cited": _fmt_ts(r[7]),
            "anchor": r[8], "recovery_status": r[9], "capture_date": _fmt_ts(r[10]),
            "capture_distance_days": r[11], "mimetype": r[12],
            # A copy from another archive (services/memento.py) cites that archive.
            "wayback_url": (r[26] if r[25] and r[26] else
                            wayback_view_url(r[10], r[1]) if recovered else wayback_view_url(None, r[1])),
            "source_archive": r[25] or ("Wayback Machine" if recovered else None),
            "local_path": r[13], "analysis_status": r[14], "title": r[15], "description": r[16],
            "content_type": r[17], "sentiment": r[18], "subjects": _json_list(r[19]),
            "tags": _json_list(r[20]), "people": _json_list(r[21]), "relation": r[22],
            "relation_note": r[23], "model": r[24], "citing": citing.get(r[0], []),
        }
        link["raw_file"] = _raw_name("linked", r[0], r[13] or "", r[12]) if recovered and r[13] else None
        links.append(link)

    # How much of the site's own writing survives at all: of the articles its
    # pages linked to, how many the Archive ever captured. On a small site the
    # crawler often took the front page and the category listings and missed
    # most individual articles, and that loss is the historian's first question.
    survival = dict((await db.execute(text("""
        SELECT recovery_status, COUNT(*) FROM link_targets
        WHERE project_id = :pid AND is_internal = 1 AND kind = 'article' GROUP BY 1
    """), {"pid": pid})).fetchall())
    articles_linked = sum(survival.values())
    articles_held = survival.get("in_corpus", 0) + survival.get("recovered", 0)
    external_ref = dict((await db.execute(text("""
        SELECT recovery_status, COUNT(*) FROM link_targets
        WHERE project_id = :pid AND is_internal = 0 AND kind IN ('reference', 'document') GROUP BY 1
    """), {"pid": pid})).fetchall())

    link_stats = dict((await db.execute(text(
        "SELECT recovery_status, COUNT(*) FROM link_targets WHERE project_id = :pid GROUP BY 1"
    ), {"pid": pid})).fetchall())
    kind_stats = (await db.execute(text(
        "SELECT is_internal, kind, COUNT(*) FROM link_targets WHERE project_id = :pid GROUP BY 1, 2"
    ), {"pid": pid})).fetchall()
    citations = (await db.execute(text(
        "SELECT COUNT(*) FROM page_links WHERE project_id = :pid"), {"pid": pid})).scalar() or 0
    models = sorted({p["model"].split(" (")[0] for p in pages if p["model"]}
                    | {l["model"] for l in links if l["model"]})

    people = sorted(people_index.values(), key=lambda e: (-len(e["pages"]), e["name"].lower()))
    # Close readings: the per-document apparatus and what it adds up to.
    # Absent on a project that has not been read yet, in which case the export
    # simply omits those pages rather than failing.
    # How evenly the Archive sampled the studied period. The point of the
    # figure is the contrast with the loss rate: a crawler can be present in
    # nearly every month and still miss most of what was published, because it
    # saves what is on the front page the day it calls.
    months_seen = sorted({t[:6] for t in timestamps if t and len(t) >= 6})
    coverage = {"months_seen": len(months_seen), "months_span": 0, "missing_months": []}
    if months_seen:
        first, last = months_seen[0], months_seen[-1]
        span = []
        year, month = int(first[:4]), int(first[4:6])
        while f"{year}{month:02d}" <= last:
            span.append(f"{year}{month:02d}")
            month = month + 1 if month < 12 else 1
            year = year if month != 1 else year + 1
        coverage["months_span"] = len(span)
        coverage["missing_months"] = [m for m in span if m not in set(months_seen)]

    app_data = await apparatus.build(db, pid)
    comment_rows = (await db.execute(text("""
        SELECT c.page_url, c.capture_timestamp, c.comment_ref, c.author, c.body,
               c.time_label, c.commented_on, c.depth, c.by_site, c.file_id, pr.title
        FROM page_comments c
        LEFT JOIN page_readings pr ON pr.file_id = c.file_id
        WHERE c.project_id = :pid
        ORDER BY c.commented_on, c.page_url, CAST(c.comment_ref AS INTEGER)
    """), {"pid": pid})).mappings().all()
    comments = [{**dict(r), "page": f"P{r['file_id']}"} for r in comment_rows]
    facts_list = await apparatus.facts(db, pid, limit=1000)
    reading_rows = (await db.execute(text("""
        SELECT pr.file_id, af.original_url, af.timestamp, pr.title, pr.genre, pr.intent,
               pr.stance, pr.secondary_intents, pr.people, pr.places, pr.organizations,
               pr.events, pr.facts, pr.frames, pr.quotes, pr.published_estimate,
               pr.period_start, pr.period_end, pr.temporal_mode, pr.summary, pr.model_used
        FROM page_readings pr JOIN archived_files af ON af.id = pr.file_id
        WHERE pr.project_id = :pid AND pr.status = 'processed' AND pr.source_file_id IS NULL
        ORDER BY af.timestamp, pr.file_id
    """), {"pid": pid})).mappings().all()
    readings = []
    for row in reading_rows:
        record = dict(row)
        for field in ("secondary_intents", "people", "places", "organizations",
                      "events", "facts", "frames", "quotes"):
            record[field] = _json_any(record.get(field))
        record["id"] = f"P{record['file_id']}"
        record["url"] = record["original_url"]
        record["date"] = _fmt_ts(record["timestamp"])
        record["wayback_url"] = wayback_view_url(record["timestamp"], record["original_url"])
        readings.append(record)

    # Images on pages that had no text (see services/images.py). One row per
    # page-image pair; the image file travels once per distinct URL.
    image_rows = (await db.execute(text("""
        SELECT pi.file_id, af.original_url AS page_url, af.timestamp AS page_timestamp,
               pi.src_url, pi.alt_text, pi.recovery_status, pi.wayback_timestamp, pi.wayback_url,
               pi.mimetype, pi.local_path, pi.description_status, pi.description,
               pi.visible_text, pi.model_used
        FROM page_images pi JOIN archived_files af ON af.id = pi.file_id
        WHERE pi.project_id = :pid
        ORDER BY af.timestamp, pi.file_id, pi.id
    """), {"pid": pid})).mappings().all()
    images = []
    for row in image_rows:
        record = dict(row)
        record["page"] = f"P{record['file_id']}"
        record["page_date"] = _fmt_ts(record["page_timestamp"])
        record["raw_file"] = (f"images/{os.path.basename(record['local_path'])}"
                              if record["local_path"] else None)
        images.append(record)

    return {
        "images": images,
        "project": project, "card": card, "findings": (card.findings if card else None) or {},
        "themes": themes, "shift": sentiment_shift(themes["years"]),
        "timeline": build_timeline(timestamps),
        "counts": {"in_scope": counts.get(1, 0), "excluded": counts.get(0, 0),
                   "pages_analyzed": len(pages), "citations": citations,
                   "links_listed": len(links),
                   "links_recovered": link_stats.get("recovered", 0),
                   "links_not_archived": link_stats.get("not_archived", 0),
                   "links_analyzed": sum(1 for l in links if l["analysis_status"] == "processed"),
                   "articles_linked": articles_linked, "articles_held": articles_held,
                   "articles_lost": survival.get("not_archived", 0),
                   "articles_lost_pct": round(100 * survival.get("not_archived", 0) / articles_linked)
                                        if articles_linked else 0,
                   "refs_recovered": external_ref.get("recovered", 0),
                   "refs_lost": external_ref.get("not_archived", 0),
                   "link_kinds": [{"scope": "internal" if i else "external", "kind": k, "count": n}
                                  for i, k, n in kind_stats]},
        "pages": pages, "links": links, "people": people, "models": models,
        "apparatus": app_data, "readings": readings, "facts": facts_list,
        "coverage": coverage,
        "comments": comments,
        "cross_check": _load_cross_check(project.domain),
        "generated_at": datetime.utcnow(),
    }


def _load_cross_check(domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The report from scripts/crosscheck_archives.py, if it has been run.

    Kept as a file written by a separate script rather than computed here: it
    needs the network and several minutes, and it answers a question about the
    Internet Archive rather than about this corpus, so it should not be redone
    every time somebody rebuilds the zip. Absent, the export simply omits the
    claims that depend on it.
    """
    path = os.path.join(DATABASE_DIR, "cross-check.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError):
        return None
    # One file per data directory, written for one site: never attach it to
    # another project's bag.
    if domain and report.get("domain") and report["domain"] != domain:
        return None
    return report


def _csv(rows: List[Dict[str, Any]], columns: List[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["; ".join(v) if isinstance(v, list) else ("" if v is None else v)
                         for v in (row.get(c) for c in columns)])
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel opens accents correctly


def _readme(project: Project, data: Dict[str, Any], bagged_on: str, models_line: str) -> bytes:
    """The note at the top of the bag: what this is, how to check it, how to cite it.

    A bag that outlives the person who made it has to say what it is without
    them. That means three things a reader cannot reconstruct from the data:
    where the material actually came from, what in here is a machine's reading
    rather than a record, and how to refer to it in a footnote.
    """
    xc = data.get("cross_check") or {}
    recovered = xc.get("recovered_elsewhere") or []
    check = ""
    if xc:
        w = xc.get("wayback_recheck", {})
        check = (
            f"\nWHAT WAS CHECKED AGAINST OTHER ARCHIVES\n"
            f"  Every article recorded here as lost was re-tested against the\n"
            f"  Wayback Machine on {xc.get('generated_at', '')[:10]}: "
            f"{w.get('lost_but_actually_have_200', 0)} turned out to be recoverable.\n"
            f"  Common Crawl, an independent crawl, was then asked for the same\n"
            f"  addresses and returned {len(recovered)}. See cross-check.json.\n")

    return (
        f"""Archive of {project.domain}
Packaged {bagged_on}.

Open data/index.html in a web browser. Nothing here needs a server or a network
connection.

WHAT THIS IS
  A reconstruction of a website that no longer exists, assembled from public
  captures held by the Internet Archive's Wayback Machine. It is not a backup
  and it was not made by the site's authors. It is what could be recovered from
  the outside, which is less than what was published -- data/methodology.html
  says how much less, and why that gap is the point.

WHAT IS EVIDENCE AND WHAT IS A READING
  The captures under data/captures/ are evidence: they are what the Archive
  holds, byte for byte. Everything else -- summaries, categories, people and
  their roles, dates, topics, findings -- is a reading produced with the help
  of a language model ({models_line}), and it contains errors. Those readings
  are finding aids. They tell you where to look. Before relying on any claim,
  open the capture it cites.
  The exact prompts the model was given are in data/prompts.txt, so the
  instrument can be inspected rather than taken on trust.
  Two parts of this archive involved no model at all: the link graph and the
  reader comments were read straight out of the captured markup.
{check}
CHECKING NOTHING WAS ALTERED
  This folder is a BagIt bag (RFC 8493). manifest-sha256.txt lists a checksum
  for every file:

      sha256sum -c manifest-sha256.txt

HOW TO CITE
  Cite the original page, not this archive, whenever you can: every page and
  link listed here carries its Wayback address, which is the citable,
  independently verifiable form and will outlive this zip.
  Cite this archive when you are referring to the reconstruction itself -- the
  counts, the distributions, the reading. Suggested form:

      Archive of {project.domain}, reconstructed from Internet Archive captures
      ({bagged_on}). BagIt package, Payload-Oxum in bag-info.txt.

  Quote the Payload-Oxum so a reader can tell which build you used.

RE-USE
  The captured pages remain the property of their authors; this package claims
  no rights over them and asserting any would be improper. The derived tables
  and readings may be re-used freely, with the caveat above about what they
  are. If you intend to republish material about named private individuals --
  reader comments especially -- consider that it was public in one context and
  is being moved into another.
""").encode("utf-8")


def write_bag(data: Dict[str, Any], include_captures: bool = True) -> str:
    """Render and zip the bag. Synchronous: run in a thread."""
    project: Project = data["project"]
    stamp = data["generated_at"].strftime("%Y%m%d-%H%M%S")
    bag_name = f"{_slug(project.domain)}-archive-{stamp}"
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    zip_path = os.path.join(EXPORTS_DIR, f"{bag_name}.zip")
    tmp_path = zip_path + ".part"

    env = _env()
    context = dict(data)
    files: Dict[str, bytes] = {}
    pages_to_render = ["index", "pages", "links", "people", "methodology"]
    # The reading pages only exist once documents have been read closely. A
    # project that has only been triaged still exports a complete, valid bag.
    if data.get("readings"):
        pages_to_render[4:4] = ["reading", "chronology"]
    for page in pages_to_render:
        files[f"{page}.html"] = env.get_template(f"{page}.html").render(**context).encode("utf-8")
    with open(os.path.join(TEMPLATES_DIR, "style.css"), "rb") as fh:
        files["style.css"] = fh.read()

    page_cols = ["id", "date", "url", "wayback_url", "content_type", "interaction_type", "sentiment",
                 "sentiment_score", "subjects", "tags", "people", "description", "estimated_date",
                 "date_confidence", "captures", "model", "raw_file"]
    link_cols = ["id", "url", "domain", "scope", "kind", "cited_by_pages", "first_cited", "last_cited",
                 "anchor", "recovery_status", "source_archive", "capture_date", "capture_distance_days",
                 "wayback_url",
                 "title", "description", "content_type", "sentiment", "relation", "relation_note",
                 "subjects", "tags", "people", "model", "raw_file"]
    files["pages.csv"] = _csv(data["pages"], page_cols)
    files["links.csv"] = _csv(data["links"], link_cols)
    files["pages.json"] = json.dumps([{k: p[k] for k in page_cols} for p in data["pages"]],
                                     ensure_ascii=False, indent=1).encode("utf-8")
    files["links.json"] = json.dumps([{**{k: l[k] for k in link_cols}, "citing": l["citing"]}
                                      for l in data["links"]], ensure_ascii=False, indent=1).encode("utf-8")
    files["themes.json"] = json.dumps(data["themes"], ensure_ascii=False, indent=1).encode("utf-8")
    if data.get("readings"):
        reading_cols = ["id", "date", "url", "wayback_url", "title", "genre", "intent", "stance",
                        "published_estimate", "period_start", "period_end", "temporal_mode",
                        "summary", "model_used"]
        files["readings.csv"] = _csv(data["readings"], reading_cols)
        files["readings.json"] = json.dumps(
            [{**{k: r[k] for k in reading_cols},
              **{k: r[k] for k in ("people", "places", "organizations", "events",
                                   "facts", "frames", "quotes", "secondary_intents")}}
             for r in data["readings"]], ensure_ascii=False, indent=1).encode("utf-8")
        files["apparatus.json"] = json.dumps(data["apparatus"], ensure_ascii=False,
                                             indent=1, default=str).encode("utf-8")
        # One row per named person per document, with the role the text gave
        # them: the flat table the voice figures are computed from, so a reader
        # can recount them without trusting this program.
        role_rows = [{"page": r["id"], "date": r["date"], "name": person.get("name"),
                      "role": person.get("role"), "affiliation": person.get("affiliation"),
                      "note": person.get("note")}
                     for r in data["readings"] for person in r["people"]]
        files["people-roles.csv"] = _csv(role_rows, ["page", "date", "name", "role",
                                                     "affiliation", "note"])
        event_rows = [{"page": r["id"], "captured": r["date"], "when": e.get("when"),
                       "what": e.get("what"), "where": e.get("where")}
                      for r in data["readings"] for e in r["events"]]
        files["events.csv"] = _csv(event_rows, ["page", "captured", "when", "what", "where"])
        if data.get("comments"):
            files["comments.csv"] = _csv(data["comments"], [
                "page", "page_url", "commented_on", "time_label", "author", "by_site",
                "depth", "body", "comment_ref", "capture_timestamp"])
        files["facts.csv"] = _csv(
            [{"page": f"P{f['file_id']}", "year": f["year"], "assertion": f["fact"]}
             for f in data["facts"]], ["page", "year", "assertion"])
    if data.get("images"):
        image_cols = ["page", "page_date", "page_url", "src_url", "alt_text", "recovery_status",
                      "wayback_timestamp", "wayback_url", "mimetype", "description_status",
                      "description", "visible_text", "model_used", "raw_file"]
        files["images.csv"] = _csv(data["images"], image_cols)
        files["images.json"] = json.dumps([{k: i[k] for k in image_cols} for i in data["images"]],
                                          ensure_ascii=False, indent=1).encode("utf-8")
    files["findings.json"] = json.dumps(data["findings"], ensure_ascii=False, indent=1).encode("utf-8")
    # The instrument itself. A method that runs on prompts is only inspectable
    # if the prompts travel with the results.
    files["prompts.txt"] = build_prompt_book(data["project"], data["models"]).encode("utf-8")
    if data.get("cross_check"):
        files["cross-check.json"] = json.dumps(data["cross_check"], ensure_ascii=False,
                                               indent=1).encode("utf-8")

    captures: List[tuple] = []
    if include_captures:
        seen_raw = set()
        for item in data["pages"] + data["links"] + data.get("images", []):
            if item.get("raw_file") and item.get("local_path") and item["raw_file"] not in seen_raw:
                src = resolve_local_path(item["local_path"])
                if src and os.path.exists(src):
                    seen_raw.add(item["raw_file"])
                    captures.append((item["raw_file"], src))

    manifest = []
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, content in files.items():
            arc = f"{bag_name}/data/{name}"
            zf.writestr(arc, content)
            manifest.append((hashlib.sha256(content).hexdigest(), f"data/{name}"))
        for name, src in captures:
            digest = hashlib.sha256()
            with open(src, "rb") as fh, zf.open(f"{bag_name}/data/{name}", "w") as out:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
            manifest.append((digest.hexdigest(), f"data/{name}"))

        payload_bytes = sum(len(c) for c in files.values()) + sum(os.path.getsize(s) for _, s in captures)
        zf.writestr(f"{bag_name}/bagit.txt", "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n")
        bagged_on = data["generated_at"].strftime("%Y-%m-%d")
        models_line = ", ".join(data["models"]) or "none recorded"
        bag_info = (
            f"Source-Organization: Digital Archeology\n"
            f"External-Description: Archive of {project.domain} recovered from the Internet Archive"
            f" Wayback Machine, analyzed with AI assistance. Open data/index.html.\n"
            f"External-Identifier: {project.domain}\n"
            f"Bagging-Date: {bagged_on}\n"
            f"Payload-Oxum: {payload_bytes}.{len(manifest)}\n"
            # Named so the bag says what produced it without being opened. The
            # prompts these models were given travel with the data, in
            # data/prompts.txt.
            f"Analysis-Models: {models_line}\n"
            f"Analysis-Prompts: data/prompts.txt\n"
        )
        zf.writestr(f"{bag_name}/bag-info.txt", bag_info)
        zf.writestr(f"{bag_name}/manifest-sha256.txt",
                    "".join(f"{h}  {p}\n" for h, p in sorted(manifest, key=lambda m: m[1])))
        zf.writestr(f"{bag_name}/README.txt", _readme(project, data, bagged_on, models_line))
    shutil.move(tmp_path, zip_path)
    return zip_path


async def build_archive(db: AsyncSession, project: Project, include_captures: bool = True) -> str:
    data = await gather(db, project)
    return await asyncio.to_thread(write_bag, data, include_captures)


def latest_export(project: Project) -> Optional[str]:
    if not os.path.isdir(EXPORTS_DIR):
        return None
    prefix = f"{_slug(project.domain)}-archive-"
    candidates = sorted(f for f in os.listdir(EXPORTS_DIR) if f.startswith(prefix) and f.endswith(".zip"))
    return os.path.join(EXPORTS_DIR, candidates[-1]) if candidates else None
