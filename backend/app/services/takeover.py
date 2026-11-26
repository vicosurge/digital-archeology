"""Did the domain change hands? Proposing the period under study.

Expired domains are bought for their inbound links and filled with spam,
gambling or a parking page. Their captures sit in the same Wayback history as
the original site and, unscoped, they swamp it (75% of b1mas4.com's captures
are a pharmacy farm). This scores each year of held captures against the
site's early years and proposes where the original site ends.

It reads the pages themselves, not the AI analyses: out-of-scope captures are
never analyzed, and the evidence has to hold for a person who opens them.
Per year, from a sample of up to SAMPLE_PER_YEAR distinct pages:

- spam: pages using several words from pharmacy, gambling, loan, replica,
  adult and essay-mill vocabularies (English and Spanish);
- parked: "this domain is for sale" and registrar parking pages;
- vocabulary: how much of the year's common vocabulary the early years
  shared (a new owner writes about other things);
- script: a switch to CJK text on a site that had none;
- generator: the CMS named in <meta name="generator">, when it changes;
- registration: an RDAP creation date after the site began means the domain
  was registered again (taken from the archival card, when one exists).

Nothing is applied. The proposal, its confidence and its evidence are shown on
the Research page, and a person applies it or not.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .llm import llm_service

SAMPLE_PER_YEAR = 30
READ_BYTES = 300_000
TOP_WORDS = 150
BASELINE_YEARS = 2
MIN_PAGES = 3          # a year with fewer sampled pages is not judged
EMPTY_WORDS = 10       # a page with fewer words is a placeholder
SUBSTANTIVE_WORDS = 30

_SPAM = [
    # pharmacy
    "viagra", "cialis", "levitra", "pharmacy", "farmacia", "prescription", "receta", "pills",
    "pastillas", "tablets", "comprimidos", "dosage", "generic", "genérico", "antibióticos",
    "antibiotics", "disfunción eréctil", "erectile", "amoxicillin", "tramadol",
    # gambling
    "casino", "poker", "slots", "betting", "apuestas", "sportsbook", "jackpot", "bet365",
    "baccarat", "roulette", "ruleta", "百家乐", "博彩", "娱乐城", "赌场", "彩票", "体育投注",
    # loans and money
    "payday", "loans", "préstamos", "forex", "crypto", "bitcoin",
    # replicas and fakes
    "replica", "réplica", "louis vuitton", "rolex", "outlet",
    # adult
    "escort", "porn", "xxx", "webcam",
    # essays
    "essay writing", "write my essay",
]
_PARKED = [
    "domain is for sale", "this domain may be for sale", "buy this domain", "domain for sale",
    "dominio está en venta", "este dominio", "is parked", "parked free", "domain parking",
    "godaddy.com", "sedo", "hugedomains", "dan.com", "afternic", "related searches",
    "the domain has expired", "this domain has expired", "renew this domain",
]
_STOP = set("""
that this with from have your what when were will there their which about would
they them then than been more also into only other some such these those very
just over most like time year page home para como pero esta este todo todos
sobre desde entre cuando donde porque también puede hace hacer tiene tienen
ellos ellas nuestro nuestra está están será sólo solo más años nbsp http https
www html index copyright rights reserved todos derechos reservados
""".split())

_GENERATOR = re.compile(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)', re.I)
_GENERATOR2 = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']generator["\']', re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_WORD = re.compile(r"[^\W\d_]{4,}", re.U)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")


def _terms(body: str, vocab: List[str]) -> List[str]:
    low = body.lower()
    hits = []
    for term in vocab:
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", low) if term.isascii() else term in low:
            hits.append(term)
    return hits


def read_page(path: str) -> Dict[str, Any]:
    """What one capture says, reduced to the signals used here."""
    from ..config import resolve_local_path
    try:
        with open(resolve_local_path(path), "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read(READ_BYTES)
    except OSError:
        return {}
    body = llm_service.clean_html(raw)
    gen = _GENERATOR.search(raw) or _GENERATOR2.search(raw)
    title = _TITLE.search(raw)
    words = [w.lower() for w in _WORD.findall(body)]
    letters = sum(1 for c in body if c.isalpha()) or 1
    return {
        "title": re.sub(r"\s+", " ", title.group(1)).strip()[:200] if title else "",
        "generator": gen.group(1).split(" ")[0].strip() if gen else "",
        "spam": _terms(body, _SPAM),
        "parked": _terms(body, _PARKED),
        "words": {w for w in words if w not in _STOP},
        "cjk": len(_CJK.findall(body)) / letters,
        "length": len(words),
    }


def score_years(by_year: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Per-year signals and a verdict, oldest first. Pure: testable."""
    years = sorted(y for y, pages in by_year.items() if any(pages))
    stats: List[Dict[str, Any]] = []
    for y in years:
        pages = [p for p in by_year[y] if p]
        n = len(pages)
        df: Counter = Counter()
        for p in pages:
            df.update(p["words"])
        stats.append({
            "year": y, "sampled": n,
            "spam_rate": sum(1 for p in pages if len(set(p["spam"])) >= 2) / n if n else 0,
            "parked_rate": sum(1 for p in pages if p["parked"]) / n if n else 0,
            "empty_rate": sum(1 for p in pages if p["length"] < EMPTY_WORDS) / n if n else 0,
            "cjk": sum(p["cjk"] for p in pages) / n if n else 0,
            "top_words": {w for w, _ in df.most_common(TOP_WORDS)},
            "generators": Counter(p["generator"] for p in pages if p["generator"]),
            "spam_terms": Counter(t for p in pages for t in set(p["spam"])),
            "titles": [p["title"] for p in pages if p["title"]][:3],
        })
    judged = [s for s in stats if s["sampled"] >= MIN_PAGES]
    baseline_years = [s for s in judged if s["spam_rate"] < 0.3 and s["parked_rate"] < 0.5][:BASELINE_YEARS]
    baseline = set().union(*(s["top_words"] for s in baseline_years)) if baseline_years else set()
    base_cjk = max((s["cjk"] for s in baseline_years), default=0)
    base_gen = set().union(*(set(s["generators"]) for s in baseline_years)) if baseline_years else set()

    for s in stats:
        s["similarity"] = (len(s["top_words"] & baseline) / len(s["top_words"])
                           if baseline and s["top_words"] else None)
        reasons = []
        if s["spam_rate"] >= 0.3:
            top = ", ".join(t for t, _ in s["spam_terms"].most_common(4))
            reasons.append(f"{s['spam_rate']:.0%} of sampled pages use spam vocabulary ({top})")
        if s["parked_rate"] >= 0.5:
            reasons.append(f"{s['parked_rate']:.0%} of sampled pages are parking or for-sale pages")
        if s["cjk"] > 0.2 and base_cjk < 0.05:
            reasons.append("the text switches to Chinese/Japanese script")
        if s["similarity"] is not None and s["similarity"] < 0.15 and s["year"] not in {b["year"] for b in baseline_years}:
            reasons.append(f"only {s['similarity']:.0%} of its common words appear in the early years")
        new_gen = set(s["generators"]) - base_gen
        if base_gen and new_gen:
            reasons.append("generator changes to " + ", ".join(sorted(new_gen)))
        s["reasons"] = reasons
        strong = sum(1 for r in reasons if not r.startswith("generator"))
        s["bad"] = s["sampled"] >= MIN_PAGES and (strong >= 1 and (s["spam_rate"] >= 0.3 or s["parked_rate"] >= 0.5
                                                                   or s["cjk"] > 0.2 or strong >= 2))
        if s["empty_rate"] >= 0.6:
            reasons.append(f"{s['empty_rate']:.0%} of sampled pages are empty placeholders")
        # Too few pages, or only placeholders: not judged either way. Between
        # the last good year and the takeover these are the lapse.
        s["thin"] = s["sampled"] < MIN_PAGES or s["empty_rate"] >= 0.6
        del s["top_words"]
        s["generators"] = dict(s["generators"])
        s["spam_terms"] = dict(s["spam_terms"].most_common(6))
    return stats


def propose(stats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The first year after which the site stays taken over, if any.

    The takeover has to stick: from that year on, at least three quarters of
    the judged years are bad, and at least one good year comes before it.
    Thin years (too few pages) between the last good year and it are counted
    with the takeover, since an expired domain is captured rarely.
    """
    judged = [s for s in stats if not s["thin"]]
    for i, s in enumerate(judged):
        if not s["bad"] or i == 0 or not any(not p["bad"] for p in judged[:i]):
            continue
        rest = judged[i:]
        if sum(1 for r in rest if r["bad"]) / len(rest) < 0.75:
            continue
        last_good = max(p["year"] for p in judged[:i] if not p["bad"])
        bad_years = [r for r in rest if r["bad"]]
        strong = sum(1 for r in bad_years if r["spam_rate"] >= 0.3 or r["parked_rate"] >= 0.5 or r["cjk"] > 0.2)
        confidence = "high" if strong >= 2 or (strong >= 1 and len(bad_years) >= 2) else "medium"
        return {"last_good_year": last_good, "first_bad_year": s["year"], "confidence": confidence,
                "gap_years": [t["year"] for t in stats if t["thin"] and last_good < t["year"] < s["year"]]}
    return None


async def _sample(db, project_id: int) -> Dict[str, List[Dict[str, Any]]]:
    rows = (await db.execute(text("""
        SELECT substr(timestamp, 1, 4) AS yr, original_url, MIN(local_path)
        FROM archived_files
        WHERE project_id = :pid AND local_path IS NOT NULL
          AND (mimetype IN ('text/html', 'application/xhtml+xml') OR mimetype IS NULL)
        GROUP BY yr, original_url
    """), {"pid": project_id})).fetchall()
    by_year: Dict[str, List[str]] = {}
    for yr, _url, path in rows:
        by_year.setdefault(yr, []).append(path)
    picked: Dict[str, List[str]] = {}
    for yr, paths in by_year.items():
        paths.sort()
        step = max(1, len(paths) // SAMPLE_PER_YEAR)
        picked[yr] = paths[::step][:SAMPLE_PER_YEAR]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for yr, paths in picked.items():
        out[yr] = list(await asyncio.gather(*(asyncio.to_thread(read_page, p) for p in paths)))
    return out


async def _boundary(db, project_id: int, last_good_year: str, first_bad_year: str) -> Dict[str, Optional[str]]:
    """The last good-looking and first bad-looking capture around the switch.

    Walks the captures from the last good year on, in time order, and stops at
    the first one that reads as spam or parking.
    """
    sql = """
        SELECT timestamp, local_path FROM archived_files
        WHERE project_id = :pid AND local_path IS NOT NULL
          AND (mimetype IN ('text/html', 'application/xhtml+xml') OR mimetype IS NULL)
          AND substr(timestamp, 1, 4) >= :y0 AND substr(timestamp, 1, 4) {op} :y1
        ORDER BY timestamp {order} LIMIT :n
    """
    # Every capture from the last good year up to the bad one (the lapse is
    # captured rarely), then the start of the bad year: the switch is there.
    before = (await db.execute(text(sql.format(op="<", order="DESC")), {
        "pid": project_id, "y0": last_good_year, "y1": first_bad_year, "n": 400})).fetchall()
    during = (await db.execute(text(sql.format(op="=", order="ASC")), {
        "pid": project_id, "y0": first_bad_year, "y1": first_bad_year, "n": 60})).fetchall()
    rows = list(reversed(before)) + list(during)
    pages = await asyncio.gather(*(asyncio.to_thread(read_page, p) for _, p in rows))
    first_bad = next((ts for (ts, _), pg in zip(rows, pages)
                      if pg and (len(set(pg["spam"])) >= 2 or pg["parked"] or pg["cjk"] > 0.2)), None)
    last_ok = None
    after_ok = None
    for (ts, _), pg in zip(rows, pages):
        if first_bad and ts >= first_bad:
            break
        if pg and pg["length"] >= SUBSTANTIVE_WORDS:
            last_ok, after_ok = ts, None
        elif last_ok and after_ok is None:
            after_ok = ts
    return {"last_good": last_ok, "first_bad": first_bad, "first_after": after_ok or first_bad}


_cache: Dict[int, tuple] = {}


async def detect_cached(db, project_id: int) -> Dict[str, Any]:
    """detect(), reused until the project's held captures change."""
    stamp = tuple((await db.execute(text(
        "SELECT COUNT(*), MAX(id) FROM archived_files WHERE project_id = :pid AND local_path IS NOT NULL"),
        {"pid": project_id})).fetchone())
    hit = _cache.get(project_id)
    if hit and hit[0] == stamp:
        return hit[1]
    result = await detect(db, project_id)
    _cache[project_id] = (stamp, result)
    return result


async def detect(db, project_id: int) -> Dict[str, Any]:
    """Score every year and, when the site changed hands, propose a scope end."""
    stats = score_years(await _sample(db, project_id))
    proposal = propose(stats)
    result: Dict[str, Any] = {"years": stats, "proposal": None}
    if not proposal:
        return result
    edge = await _boundary(db, project_id, proposal["last_good_year"], proposal["first_bad_year"])
    # Scope ends at the first capture that reads as the new owner, or, failing
    # that, at the start of the first bad year.
    end = edge["first_after"] or f"{proposal['first_bad_year']}0101000000"
    evidence = []
    for s in stats:
        if s["bad"] and s["year"] >= proposal["first_bad_year"]:
            evidence.append(f"{s['year']} ({s['sampled']} pages sampled): " + "; ".join(s["reasons"]))
    card = (await db.execute(text(
        "SELECT whois_created FROM archival_cards WHERE project_id = :pid"), {"pid": project_id})).scalar()
    first_year = stats[0]["year"] if stats else None
    if card and first_year and str(card)[:4] > first_year:
        evidence.append(f"RDAP says the domain was registered on {str(card)[:10]}, after the site began "
                        f"({first_year}): it lapsed and was registered again.")
        if proposal["confidence"] == "medium":
            proposal["confidence"] = "high"
    result["proposal"] = {**proposal, **edge, "scope_end": end[:14].ljust(14, "0"),
                          "evidence": evidence,
                          "reason": _reason(proposal, edge, stats)}
    return result


def _reason(p: Dict[str, Any], edge: Dict[str, Optional[str]], stats: List[Dict[str, Any]]) -> str:
    bad = [s for s in stats if s["bad"] and s["year"] >= p["first_bad_year"]]
    kinds = []
    if any(s["spam_rate"] >= 0.3 for s in bad):
        terms = Counter()
        for s in bad:
            terms.update(s["spam_terms"])
        kinds.append("spam (" + ", ".join(t for t, _ in terms.most_common(3)) + ")")
    if any(s["parked_rate"] >= 0.5 for s in bad):
        kinds.append("a parking page")
    if any(s["cjk"] > 0.2 for s in bad):
        kinds.append("Chinese-language content")
    what = " and ".join(kinds) or "unrelated content"

    def ym(ts: Optional[str]) -> str:
        return f"{ts[:4]}-{ts[4:6]}" if ts else ""

    last = edge.get("last_good")
    parts = [f"The original site's last substantive capture dates from {ym(last) or p['last_good_year']}."]
    after, bad = edge.get("first_after"), edge.get("first_bad")
    if after and bad and after < bad:
        parts.append(f"From {ym(after)} its pages are empty, and by {ym(bad)} the domain had lapsed.")
    elif bad and bad[:4] < p["first_bad_year"]:
        parts.append(f"By {ym(bad)} the domain had lapsed.")
    parts.append(f"From {p['first_bad_year']} it serves {what}. Later captures are kept as evidence of the "
                 "takeover but not analyzed.")
    return " ".join(parts)
