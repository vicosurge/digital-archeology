"""Aggregates over the close readings: the critical apparatus of the archive.

services/reading.py reads one document at a time. This module is what those
readings add up to, and it is where the borrowed methods actually live:

- **Actors and voice.** Counting people is thin. Counting them *by the role the
  text gives them* is a claim about the publication: who it lets speak, and who
  it only speaks about. ``voice_asymmetry`` reports exactly that gap. This is
  the standard sociological question about a media outlet -- access to the
  means of speech -- asked of the corpus rather than asserted about it.

- **Masthead detection.** Sidebars, bylines and staff boxes repeat on every
  page, so the most "frequent" person in a naive count is whoever edits the
  site. Anyone appearing on more than ``MASTHEAD_SHARE`` of documents is
  treated as furniture and reported separately, the same trick the link
  classifier uses for sitewide banners. The masthead is a finding in itself:
  it is the editorial team, recovered without anyone writing it down.

- **Geography.** ``gazetteer`` types and counts places, so "the site covered
  the border region" stops being an impression and becomes a distribution with
  a tail worth explaining.

- **Two clocks.** ``chronology`` separates when a piece was published from when
  the events it discusses happened, and ``temporal_reach`` measures the gap.
  A publication whose pieces increasingly look backwards is doing something
  different from one reporting yesterday, and the capture date -- the only date
  most web archaeology has -- cannot tell them apart.

- **Purpose over time.** ``purpose_mix`` tracks genre, intent and stance per
  year. Whether a project achieved what it set out to do is, in part, whether
  it kept doing the same kind of thing.

- **Frames.** Recurring construals of a subject, in the source's own wording.
  This is the discourse-analytic residue: not what was covered, but under what
  description.

Everything here reads only directly-read rows (``source_file_id IS NULL``);
copies made for duplicate captures would multiply every count.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .reception import page_key

# A person on more than this share of read documents is site furniture
# (masthead, sidebar, comment widget), not a subject of coverage.
MASTHEAD_SHARE = 0.25

# Initials and stray tokens ("JE", "N.") are not people. Anything this short is
# dropped rather than guessed at.
MIN_NAME_LENGTH = 3

# Particles that carry no identifying weight when matching name variants.
_PARTICLES = {"de", "del", "la", "las", "los", "y", "e", "da", "do", "van", "von", "el"}


def fold(value: str) -> str:
    """Accent- and case-insensitive key. 'César Faz' and 'Cesar Faz' are one man."""
    stripped = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(stripped.lower().replace(".", " ").split())


def _tokens(folded: str) -> frozenset:
    return frozenset(t for t in folded.split() if t not in _PARTICLES)


def merge_variants(keys: List[str]) -> Dict[str, str]:
    """Map each folded name onto the fullest spelling of the same person.

    The rule is deliberately conservative and stated so it can be checked: a
    name merges into a longer one only when every one of its words appears in
    that longer name ("Peña Nieto" into "Enrique Peña Nieto"), it has at least
    one word that is not a bare particle, and exactly one longer candidate
    matches. Ambiguity is left alone -- two different Lópezes must stay two
    people, and an archive that quietly fuses them is worse than one that does
    not try.

    Nicknames and handles ("AMLO", "@EPN", "Peje") share no words with the full
    name and are never merged. They stay separate entries, which is honest:
    deciding they are the same person is the historian's call, not a string
    comparison's.
    """
    unique = set(keys)
    tokens = {k: _tokens(k) for k in unique}
    # Inverted index: the names that contain a word. A candidate must contain
    # every word of the key, so intersecting postings finds them without
    # comparing every pair (1.5k names took 3 s that way).
    postings: Dict[str, set] = {}
    for k, toks in tokens.items():
        for t in toks:
            postings.setdefault(t, set()).add(k)
    canonical: Dict[str, str] = {}
    for key in unique:
        key_tokens = tokens[key]
        if not key_tokens:
            canonical[key] = key
            continue
        lists = sorted((postings[t] for t in key_tokens), key=len)
        found = set(lists[0]).intersection(*lists[1:])
        matches = [c for c in found if c != key and key_tokens < tokens[c]]
        canonical[key] = matches[0] if len(matches) == 1 else key
    # Follow chains so "Peña" -> "Peña Nieto" -> "Enrique Peña Nieto".
    resolved: Dict[str, str] = {}
    for key in canonical:
        seen, target = {key}, canonical[key]
        while target != canonical.get(target, target) and canonical[target] not in seen:
            seen.add(target)
            target = canonical[target]
        resolved[key] = target
    return resolved


_READ = """
    FROM page_readings pr JOIN archived_files af ON af.id = pr.file_id
    WHERE pr.project_id = :pid AND pr.status = 'processed' AND pr.source_file_id IS NULL
"""


async def _json_rows(db: AsyncSession, project_id: int, column: str,
                     fields: Dict[str, str], extra: str = "") -> List[Dict[str, Any]]:
    """Explode a JSON array column into one row per element.

    ``fields`` maps output key to a JSON path inside the element. ``yr`` (the
    capture year) and ``file_id`` come along for free, so callers can group by
    time or count distinct documents.
    """
    # Two shapes of array. For arrays of objects each element is JSON and the
    # path picks a field out of it; for arrays of plain strings (facts, frames)
    # the element *is* the value, and calling json_extract on it would try to
    # parse the bare word as a document. Aliases are quoted because "when" and
    # "where" are good field names in the reading schema and reserved in SQL.
    selects = ", ".join(
        (f'je.value AS "{name}"' if path == "$"
         else f'json_extract(je.value, \'{path}\') AS "{name}"')
        for name, path in fields.items())
    rows = (await db.execute(text(f"""
        SELECT pr.file_id AS file_id, substr(af.timestamp, 1, 4) AS yr, {selects}
        FROM page_readings pr JOIN archived_files af ON af.id = pr.file_id,
             json_each(pr.{column}) je
        WHERE pr.project_id = :pid AND pr.status = 'processed'
          AND pr.source_file_id IS NULL {extra}
    """), {"pid": project_id})).mappings().all()
    return [dict(r) for r in rows]


async def document_count(db: AsyncSession, project_id: int) -> int:
    return (await db.execute(text(f"SELECT COUNT(*) {_READ}"), {"pid": project_id})).scalar() or 0


# ===================== actors =====================

def _is_furniture(person: Dict[str, Any]) -> bool:
    """Is this name part of the page, or part of what the page is about?

    Frequency alone cannot tell the two apart, and assuming it can produced a
    genuinely wrong answer: across b1mas4 the editor appeared on 27% of
    documents and Enrique Pena Nieto on 26%, so a bare threshold filed a
    sitting president as newsroom furniture.

    The role breaks the tie. Furniture appears in one unchanging capacity --
    a byline block, a staff box, a sidebar -- while a subject of coverage is
    written *about*. So a frequent name is furniture when either:

    - writing is what it mostly does (a byline; the editor was author on 185
      of his 201 appearances, the president on none of his 199), or
    - it is nearly ubiquitous and never anything but incidental, which is what
      a staff box listing the whole team on every page looks like.

    Everything else is coverage, however often it recurs.
    """
    if person["share"] < MASTHEAD_SHARE:
        return False
    appearances = sum(person["roles"].values()) or 1
    if person["authored"] / appearances >= 0.5:
        return True
    incidental = person["roles"].get("mentioned", 0)
    return person["share"] >= 0.6 and incidental / appearances >= 0.9


async def _commenter_keys(db: AsyncSession, project_id: int) -> set:
    """Folded names of everyone who left a comment, from the markup."""
    rows = (await db.execute(text(
        "SELECT DISTINCT author FROM page_comments WHERE project_id = :pid AND author IS NOT NULL"
    ), {"pid": project_id})).scalars().all()
    return {fold(a) for a in rows if a and len(a.strip()) >= MIN_NAME_LENGTH}


async def actors(db: AsyncSession, project_id: int, limit: int = 40) -> Dict[str, Any]:
    """People, split by the role the text gives them, with the masthead pulled out.

    Readers are removed here. A model reading a captured page sees the comment
    thread as part of the document and reports the people arguing in it as
    named in passing, which quietly turns an audience into coverage. The
    comment markup says exactly who those people are, so anyone who only ever
    appears in passing *and* left a comment is moved to the reception side. The
    guard matters: a politician who is written about and also happens to have
    commented stays an actor, because their other roles prove it.
    """
    total = await document_count(db, project_id)
    rows = await _json_rows(db, project_id, "people",
                            {"name": "$.name", "role": "$.role",
                             "affiliation": "$.affiliation", "note": "$.note"})
    commenters = await _commenter_keys(db, project_id)

    # Fold case and accents for counting but keep the most common spelling for
    # display, then merge partial namings into the fullest form (see
    # merge_variants) so one person is not counted as three.
    # The project's canonicalizer applies the same automatic rule plus any
    # merges the historian made on the People page (services/people.py).
    from .people import canonicalizer
    canon = await canonicalizer(db, project_id,
                                extra=[(r.get("name") or "").strip() for r in rows])
    commenters = {canon.canonical(c) for c in commenters}

    docs: Dict[str, set] = defaultdict(set)
    roles: Dict[str, Counter] = defaultdict(Counter)
    spellings: Dict[str, Counter] = defaultdict(Counter)
    affiliations: Dict[str, Counter] = defaultdict(Counter)
    years: Dict[str, set] = defaultdict(set)
    for row in rows:
        name = (row.get("name") or "").strip()
        if len(name) < MIN_NAME_LENGTH:
            continue
        key = canon.canonical(name)
        docs[key].add(row["file_id"])
        roles[key][row.get("role") or "mentioned"] += 1
        spellings[key][name] += 1
        if row.get("affiliation"):
            affiliations[key][str(row["affiliation"]).strip()] += 1
        if row.get("yr"):
            years[key].add(row["yr"])

    people = []
    audience = []
    for key, file_ids in docs.items():
        n = len(file_ids)
        counts = roles[key]
        if key in commenters and not (counts.get("author") or counts.get("quoted")
                                      or counts.get("subject")):
            audience.append(spellings[key].most_common(1)[0][0])
            continue
        people.append({
            "name": spellings[key].most_common(1)[0][0],
            "documents": n,
            "share": n / total if total else 0.0,
            "roles": dict(roles[key]),
            "authored": roles[key].get("author", 0),
            "quoted": roles[key].get("quoted", 0),
            "subject": roles[key].get("subject", 0),
            "affiliation": (affiliations[key].most_common(1)[0][0] if affiliations[key] else None),
            "years": sorted(years[key]),
        })
    people.sort(key=lambda p: (-p["documents"], p["name"]))

    masthead = [p for p in people if _is_furniture(p)]
    covered = [p for p in people if not _is_furniture(p)]
    return {
        "documents": total,
        "masthead": masthead,
        "authors": sorted([p for p in covered if p["authored"]],
                          key=lambda p: -p["authored"])[:limit],
        "quoted": sorted([p for p in covered if p["quoted"]],
                         key=lambda p: -p["quoted"])[:limit],
        "subjects": sorted([p for p in covered if p["subject"]],
                           key=lambda p: -p["subject"])[:limit],
        "all": covered[:limit],
        "distinct_people": len(people),
        "moved_to_audience": sorted(audience),
    }


def voice_asymmetry(actor_data: Dict[str, Any], limit: int = 12) -> Dict[str, Any]:
    """Who is given speech, and who is only spoken about.

    Two lists, both drawn from the same population: people the publication
    quotes but rarely writes about, and people it writes about but never
    quotes. The second list is the interesting one -- it is a record of who the
    outlet treated as a topic rather than a voice.
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for bucket in ("quoted", "subjects", "authors"):
        for person in actor_data.get(bucket, []):
            by_name[person["name"]] = person
    # Every person with both counts lands in exactly one list, by which side is
    # heavier. An earlier version required "never quoted at all" for the second
    # list, which silently dropped the most telling cases: a candidate written
    # about fifteen times and quoted once belongs there, not nowhere.
    speakers, spoken_about = [], []
    for person in by_name.values():
        q, s = person["quoted"], person["subject"]
        if q and q > s:
            speakers.append(person)
        elif s and s > q:
            spoken_about.append(person)
    speakers.sort(key=lambda p: (-(p["quoted"] - p["subject"]), -p["quoted"]))
    spoken_about.sort(key=lambda p: (-(p["subject"] - p["quoted"]), -p["subject"]))
    return {"speakers": speakers[:limit], "spoken_about": spoken_about[:limit]}


# ===================== geography =====================

async def gazetteer(db: AsyncSession, project_id: int, limit: int = 40) -> Dict[str, Any]:
    rows = await _json_rows(db, project_id, "places",
                            {"name": "$.name", "kind": "$.kind", "note": "$.note"})
    docs: Dict[str, set] = defaultdict(set)
    kinds: Dict[str, Counter] = defaultdict(Counter)
    spellings: Dict[str, Counter] = defaultdict(Counter)
    by_year: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        key = fold(name)
        docs[key].add(row["file_id"])
        kinds[key][row.get("kind") or "other"] += 1
        spellings[key][name] += 1
        if row.get("yr"):
            by_year[row["yr"]][key] += 1

    places = [{
        "name": spellings[key].most_common(1)[0][0],
        "kind": kinds[key].most_common(1)[0][0],
        "documents": len(file_ids),
    } for key, file_ids in docs.items()]
    places.sort(key=lambda p: (-p["documents"], p["name"]))

    kind_totals = Counter(p["kind"] for p in places)
    return {
        "places": places[:limit],
        "distinct_places": len(places),
        "by_kind": kind_totals.most_common(),
        "by_year": {yr: [(spellings[k].most_common(1)[0][0], n) for k, n in c.most_common(6)]
                    for yr, c in sorted(by_year.items())},
    }


async def organizations(db: AsyncSession, project_id: int, limit: int = 40) -> Dict[str, Any]:
    rows = await _json_rows(db, project_id, "organizations", {"name": "$.name", "kind": "$.kind"})
    docs: Dict[str, set] = defaultdict(set)
    kinds: Dict[str, Counter] = defaultdict(Counter)
    spellings: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        key = fold(name)
        docs[key].add(row["file_id"])
        kinds[key][row.get("kind") or "other"] += 1
        spellings[key][name] += 1
    orgs = [{"name": spellings[k].most_common(1)[0][0],
             "kind": kinds[k].most_common(1)[0][0],
             "documents": len(v)} for k, v in docs.items()]
    orgs.sort(key=lambda o: (-o["documents"], o["name"]))
    return {"organizations": orgs[:limit], "distinct_organizations": len(orgs),
            "by_kind": Counter(o["kind"] for o in orgs).most_common()}


# ===================== time =====================

def _year_of(value: Optional[str]) -> Optional[str]:
    """First 4-digit year in a loose date string, if it is a plausible one."""
    if not value:
        return None
    digits = ""
    for ch in str(value):
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                break
        else:
            digits = ""
    if len(digits) == 4 and 1800 <= int(digits) <= 2100:
        return digits
    return None


async def chronology(db: AsyncSession, project_id: int, limit: int = 200) -> Dict[str, Any]:
    """Events the corpus reports, dated by when they happened.

    An event list assembled this way is not a timeline of the site; it is a
    timeline of what the site considered worth reporting, which is the thing a
    historian is usually after.
    """
    rows = await _json_rows(db, project_id, "events",
                            {"what": "$.what", "when": "$.when", "where": "$.where"})
    events = []
    for row in rows:
        what = (row.get("what") or "").strip()
        if not what:
            continue
        events.append({
            "what": what,
            "when": (row.get("when") or "").strip() or None,
            "where": (row.get("where") or "").strip() or None,
            "event_year": _year_of(row.get("when")),
            "capture_year": row.get("yr"),
            "file_id": row["file_id"],
        })
    dated = [e for e in events if e["event_year"]]
    dated.sort(key=lambda e: (e["event_year"], e["what"]))
    return {
        "events": dated[:limit],
        "total_events": len(events),
        "dated_events": len(dated),
        "by_year": Counter(e["event_year"] for e in dated).most_common(),
    }


async def temporal_reach(db: AsyncSession, project_id: int) -> Dict[str, Any]:
    """How far back the corpus looks, and in what mode, per capture year."""
    rows = (await db.execute(text(f"""
        SELECT substr(af.timestamp, 1, 4) AS yr, pr.temporal_mode, pr.period_start,
               pr.published_estimate
        {_READ}
    """), {"pid": project_id})).fetchall()
    modes: Dict[str, Counter] = defaultdict(Counter)
    lags: Dict[str, List[int]] = defaultdict(list)
    for yr, mode, period_start, published in rows:
        if not yr:
            continue
        modes[yr][mode or "unknown"] += 1
        start_year = _year_of(period_start)
        base_year = _year_of(published) or yr
        if start_year and base_year:
            lag = int(base_year) - int(start_year)
            if 0 <= lag <= 200:
                lags[yr].append(lag)
    return {yr: {
        "modes": dict(counter),
        "median_lookback_years": (sorted(lags[yr])[len(lags[yr]) // 2] if lags.get(yr) else None),
        "documents": sum(counter.values()),
    } for yr, counter in sorted(modes.items())}


# ===================== purpose =====================

async def purpose_mix(db: AsyncSession, project_id: int) -> Dict[str, Any]:
    """Genre, intent and stance, overall and per capture year."""
    out: Dict[str, Any] = {}
    per_year: Dict[str, Dict[str, List]] = defaultdict(dict)
    for field in ("genre", "intent", "stance"):
        rows = (await db.execute(text(f"""
            SELECT substr(af.timestamp, 1, 4) AS yr, pr.{field}, COUNT(*)
            {_READ} AND pr.{field} IS NOT NULL
            GROUP BY yr, pr.{field} ORDER BY yr, COUNT(*) DESC
        """), {"pid": project_id})).fetchall()
        totals: Counter = Counter()
        for yr, value, n in rows:
            totals[value] += n
            per_year[yr].setdefault(field, []).append((value, int(n)))
        out[field] = totals.most_common()

    secondary = await _json_rows(db, project_id, "secondary_intents", {"value": "$"})
    out["secondary_intents"] = Counter(
        (r.get("value") or "").strip() for r in secondary if r.get("value")).most_common(10)
    out["by_year"] = {yr: data for yr, data in sorted(per_year.items())}
    return out


async def frames(db: AsyncSession, project_id: int, limit: int = 30) -> List[tuple]:
    rows = await _json_rows(db, project_id, "frames", {"value": "$"})
    counts: Counter = Counter()
    spellings: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        value = (row.get("value") or "").strip()
        if not value:
            continue
        counts[value.lower()] += 1
        spellings[value.lower()][value] += 1
    return [(spellings[k].most_common(1)[0][0], n) for k, n in counts.most_common(limit)]


async def facts(db: AsyncSession, project_id: int, limit: int = 300) -> List[Dict[str, Any]]:
    """Every checkable assertion, with the document it came from.

    Deliberately unfiltered and unranked. This is the working list a historian
    verifies against other sources; ordering it by anything would be the
    machine deciding what matters.
    """
    rows = await _json_rows(db, project_id, "facts", {"value": "$"})
    out = []
    seen: set = set()
    for row in rows:
        value = (row.get("value") or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append({"fact": value, "file_id": row["file_id"], "year": row.get("yr")})
        if len(out) >= limit:
            break
    return out


# ===================== reception =====================



async def reception(db: AsyncSession, project_id: int, limit: int = 25) -> Dict[str, Any]:
    """What came back: who commented, how often, and whether the site answered.

    The headline number is the engagement rate, and it is worth stating plainly
    even -- especially -- when it is low. A publication that set out to become a
    reference for its region and drew comments on one article in fifteen has
    told you something about whether it got there, and no amount of reading its
    own prose would have revealed it.

    ``answered_share`` is the other half: a comment thread is only a
    conversation if someone replies. Comments posted from the site's own
    account are marked in the markup, so this is counted, not estimated.
    """
    rows = (await db.execute(text("""
        SELECT c.author, c.body, c.commented_on, c.by_site, c.depth,
               c.page_url, c.page_key
        FROM page_comments c WHERE c.project_id = :pid
    """), {"pid": project_id})).mappings().all()
    if not rows:
        return {"comments": 0}

    # Comments are read from the last capture of a URL and readings from the
    # representative one, so the two never share a file_id. Match on the URL
    # instead, or every "most discussed" link points at a page that is not in
    # the export.
    read_rows = (await db.execute(text("""
        SELECT af.original_url, pr.file_id, pr.title
        FROM page_readings pr JOIN archived_files af ON af.id = pr.file_id
        WHERE pr.project_id = :pid AND pr.status = 'processed' AND pr.source_file_id IS NULL
    """), {"pid": project_id})).fetchall()
    read_by_url = {page_key(url): (fid, title) for url, fid, title in read_rows}

    documents = await document_count(db, project_id)
    pages: Dict[str, Dict[str, Any]] = {}
    commenters: Dict[str, Counter] = defaultdict(Counter)
    spellings: Dict[str, Counter] = defaultdict(Counter)
    by_year: Counter = Counter()
    site_comments = 0

    for row in rows:
        # "/slug" and "/slug/" are the same article; without folding them the
        # most-discussed list shows one piece twice and splits its count.
        url = row["page_key"]
        file_id, title = read_by_url.get(url, (None, None))
        entry = pages.setdefault(url, {
            "url": row["page_url"], "title": title, "file_id": file_id,
            "comments": 0, "from_readers": 0, "answered": False})
        entry["comments"] += 1
        if row["by_site"]:
            site_comments += 1
            entry["answered"] = True
        else:
            entry["from_readers"] += 1
            if row["commented_on"]:
                by_year[row["commented_on"][:4]] += 1
            name = (row["author"] or "").strip()
            if len(name) >= MIN_NAME_LENGTH:
                commenters[fold(name)][url] += 1
                spellings[fold(name)][name] += 1

    reader_comments = sum(p["from_readers"] for p in pages.values())
    discussed = [p for p in pages.values() if p["from_readers"]]
    answered = [p for p in discussed if p["answered"]]

    voices = [{"name": spellings[k].most_common(1)[0][0],
               "comments": sum(c.values()), "pages": len(c)}
              for k, c in commenters.items()]
    voices.sort(key=lambda v: (-v["comments"], v["name"]))
    returning = [v for v in voices if v["pages"] > 1]

    return {
        "comments": len(rows),
        "reader_comments": reader_comments,
        "site_comments": site_comments,
        "pages_with_comments": len(discussed),
        "documents": documents,
        "engagement_pct": round(100 * len(discussed) / documents, 1) if documents else 0,
        "answered_pages": len(answered),
        "answered_share": round(100 * len(answered) / len(discussed)) if discussed else 0,
        "unread_pages": sum(1 for p in pages.values() if p["file_id"] is None and p["from_readers"]),
        "distinct_commenters": len(voices),
        "returning_commenters": len(returning),
        "voices": voices[:limit],
        "by_year": sorted(by_year.items()),
        "most_discussed": sorted(discussed, key=lambda p: -p["from_readers"])[:limit],
    }


async def build(db: AsyncSession, project_id: int) -> Dict[str, Any]:
    """Everything above, in one call, for the research page and the export."""
    actor_data = await actors(db, project_id)
    return {
        "documents": actor_data["documents"],
        "actors": actor_data,
        "voice": voice_asymmetry(actor_data),
        "gazetteer": await gazetteer(db, project_id),
        "organizations": await organizations(db, project_id),
        "chronology": await chronology(db, project_id),
        "temporal": await temporal_reach(db, project_id),
        "purpose": await purpose_mix(db, project_id),
        "frames": await frames(db, project_id),
        "reception": await reception(db, project_id),
    }
