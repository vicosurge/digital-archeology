"""Test the "lost" claim against archives other than the Wayback Machine.

Everything this project reports about loss is measured against one archive.
"Never captured" therefore means "not in the Internet Archive", which is a
weaker statement than it sounds, and the weakness is load-bearing: the headline
figure -- what share of its own articles the site did not keep -- rests on it.

This script attacks that figure from two directions.

1. False negatives. The recovery run happened while the Wayback CDX API was
   intermittently down, so an article could have been recorded as lost because
   a query failed rather than because nothing was kept. Instead of re-querying
   each URL, it pulls the domain's entire capture index in two requests and
   checks the lost set against it offline. Cheap, and it re-tests every URL
   rather than a sample.

2. A second archive. Common Crawl is an independent crawl with its own
   frontier, so where it overlaps it is genuine corroboration and where it
   does not, it is a genuine second chance. Its index is queried per crawl
   collection, and any recovered article is fetched out of the WARC by byte
   range so the result is the actual page, not just an index entry.

Two other archives were tried and could not be reached. They are recorded in
the output rather than dropped, because "we did not check" and "we checked and
it was unavailable" are different claims:

  archive.today -- TCP timeout to every mirror (.ph/.is/.today/.li) from this
  network. DNS resolves, port 443 never opens.
  Memento TimeTravel -- the aggregator that used to federate this query across
  all of them is gone; timetravel.mementoweb.org is NXDOMAIN globally.

Usage::

    python scripts/crosscheck_archives.py --project 4 [--db PATH] [--out PATH]

Writes a JSON report the export embeds and the methodology page cites. Needs
network. Takes a few minutes, most of it waiting politely between Common Crawl
collections.
"""
import argparse
import gzip
import html
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

UA = "digital-archeology/1.0 (archival research; one-off cross-check)"
CDX = "http://web.archive.org/cdx/search/cdx"
CC_COLLINFO = "https://index.commoncrawl.org/collinfo.json"
CC_DATA = "https://data.commoncrawl.org/"

# Crawls to ask Common Crawl about. Generous on both sides of the studied
# period: a crawl is named for when it ran, not for when the page was written.
CC_YEARS = {"2012", "2013", "2014", "2015", "2016", "2017"}

# Archives that could not be reached, kept as evidence of the attempt.
UNREACHABLE = [
    {"archive": "archive.today",
     "endpoints": ["archive.ph", "archive.is", "archive.today", "archive.li"],
     "result": "TCP connect to :443 times out from this network; DNS resolves "
               "(104.36.80.106). Not checked -- no conclusion either way."},
    {"archive": "Memento TimeTravel aggregator",
     "endpoints": ["timetravel.mementoweb.org"],
     "result": "NXDOMAIN globally; the aggregator has been retired. Would have "
               "federated this query across ~20 archives at once."},
]


CACHE_DIR = None  # set from --cache-dir


def _cache_path(url):
    if not CACHE_DIR:
        return None
    import hashlib
    return os.path.join(CACHE_DIR, hashlib.sha256(url.encode()).hexdigest()[:24] + ".cache")


def get(url, timeout=120, retries=4, backoff=15):
    """Fetch, with a disk cache and patient backoff.

    Common Crawl rate-limits by refusing the connection outright rather than
    answering 429, and it does so for a while once tripped. Since a full run
    asks it about forty-odd crawls, a re-run for any reason -- a bug in the
    parsing, a second project -- would otherwise be blocked for no good reason.
    Responses are therefore cached on disk when --cache-dir is given, and a
    404 is cached too: "nothing here" is an answer worth keeping.
    """
    path = _cache_path(url)
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            body = fh.read()
        return None if body == b"\x00__404__" else body

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            body = urllib.request.urlopen(req, timeout=timeout).read()
            break
        except urllib.error.HTTPError as e:
            # The CDX and CC index APIs both answer "nothing here" with a 404.
            # That is an answer, not a failure, so hand it back rather than retry.
            if e.code == 404:
                body = None
                break
            last = e
        except Exception as e:
            last = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    else:
        raise RuntimeError(f"giving up on {url}: {last}")

    if path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"\x00__404__" if body is None else body)
    return body


def norm(url):
    """Comparison key: scheme, www, :80 and trailing slash carry no meaning here."""
    u = (url or "").strip().lower().rstrip("/")
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return re.sub(r":80(?=/|$)", "", u)


def unsurt(key):
    """CDX sorts keys host-reversed: ``com,b1mas4)/-/foo`` -> ``b1mas4.com/-/foo``."""
    if ")" not in key:
        return key
    host, path = key.split(")", 1)
    return norm(".".join(reversed(host.split(","))) + path)


def cdx_keys(domain, **params):
    q = {"url": domain, "matchType": "domain", "output": "json",
         "fl": "urlkey", "collapse": "urlkey", "limit": "200000"}
    q.update(params)
    body = get(f"{CDX}?{urllib.parse.urlencode(q)}", timeout=300)
    if not body:
        return set()
    rows = json.loads(body.decode("utf-8", "replace"))
    return {unsurt(r[0]) for r in rows[1:]}  # row 0 is the header


def common_crawl(domain):
    colls = json.loads(get(CC_COLLINFO).decode("utf-8"))
    want = [c for c in colls if c["id"].split("-")[2][:4] in CC_YEARS]
    want.sort(key=lambda c: c["id"])
    hits, asked = [], []
    for c in want:
        q = urllib.parse.urlencode({"url": f"{domain}/*", "output": "json",
                                    "limit": "2000"})
        body = get(f"https://index.commoncrawl.org/{c['id']}-index?{q}", timeout=90)
        found = 0
        if body:
            for line in body.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "url" not in rec:  # {"message": "No Captures found..."}
                    continue
                rec["collection"] = c["id"]
                hits.append(rec)
                found += 1
        asked.append({"collection": c["id"], "records": found})
        print(f"  {c['id']:<20} {found}", file=sys.stderr)
        time.sleep(1)
    return hits, asked


def fetch_warc(rec, dest_dir):
    """Pull one record out of Common Crawl by byte range and save the response body."""
    off, length = int(rec["offset"]), int(rec["length"])
    req = urllib.request.Request(
        CC_DATA + rec["filename"],
        headers={"User-Agent": UA, "Range": f"bytes={off}-{off + length - 1}"})
    raw = gzip.decompress(urllib.request.urlopen(req, timeout=180).read())
    text = raw.decode("utf-8", "replace")
    # WARC header, blank line, HTTP header, blank line, then the body.
    parts = text.split("\r\n\r\n", 2)
    body = parts[2] if len(parts) > 2 else text
    slug = re.sub(r"[^a-z0-9]+", "-", norm(rec["url"]).split("/")[-1])[:80] or "page"
    name = f"{slug}-cc{rec['timestamp']}.txt"
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, name), "w", encoding="utf-8") as fh:
        fh.write(body)
    return {"file": name, "bytes": len(body), "title": _title(text)}


def _title(text):
    """The piece's own title, out of the <title> tag the theme built.

    WordPress themes prefix the site name ("b1mas4 &raquo; Some headline"), and
    the tag is HTML-escaped. Both would otherwise travel into the export and be
    quoted as if they were the headline.
    """
    m = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
    if not m:
        return None
    t = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    # Drop a leading "Site name <sep>" where the separator is a typographic one,
    # never a hyphen or colon that could belong to the headline itself.
    parts = re.split(r"\s*[»|]\s*", t, maxsplit=1)
    return (parts[1].strip() if len(parts) == 2 and parts[1].strip() else t) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--db", default=os.path.join(os.environ.get("DA_DATA_DIR", "data"),
                                                 "digital_archeology.db"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--recovered-dir", default=None,
                    help="where to save pages a second archive still holds")
    ap.add_argument("--cache-dir", default=None,
                    help="cache archive responses here so a re-run is free "
                         "(Common Crawl rate-limits hard once tripped)")
    args = ap.parse_args()

    global CACHE_DIR
    CACHE_DIR = args.cache_dir

    db = sqlite3.connect(args.db)
    domain, scope_end = db.execute(
        "SELECT domain, scope_end FROM projects WHERE id=?", (args.project,)).fetchone()
    to_ts = (scope_end or "20170101")[:8]

    lost = {}
    for uk, url, cites in db.execute(
            """SELECT url_key, url, cited_by_pages FROM link_targets
               WHERE project_id=? AND is_internal=1 AND kind='article'
                 AND recovery_status='not_archived'""", (args.project,)):
        lost[norm(uk)] = {"url": url, "cited_by_pages": cites or 0}
    held = {norm(u) for (u,) in db.execute(
        """SELECT original_url FROM archived_files
           WHERE project_id=? AND in_scope=1""", (args.project,))}
    print(f"{len(lost)} articles marked lost; {len(held)} captures held", file=sys.stderr)

    # 1. Re-test every lost URL against the Wayback Machine as it stands today.
    print("querying Wayback CDX...", file=sys.stderr)
    ok200 = cdx_keys(domain, filter="statuscode:200", to=to_ts)
    anycap = cdx_keys(domain)
    has_200 = sorted(k for k in lost if k in ok200)
    no_200 = sorted(k for k in lost if k not in ok200 and k in anycap)
    absent = sorted(k for k in lost if k not in anycap)

    # 2. Ask an archive with a different frontier.
    print("querying Common Crawl...", file=sys.stderr)
    cc_hits, cc_asked = common_crawl(domain)
    cc_200 = {}
    for r in cc_hits:
        if r.get("status") == "200":
            cc_200.setdefault(norm(r["url"]), []).append(r)

    recovered = []
    for key in sorted(cc_200, key=lambda k: -lost.get(k, {}).get("cited_by_pages", 0)):
        if key not in lost:
            continue
        best = sorted(cc_200[key], key=lambda r: r["timestamp"])[0]
        entry = {"url": lost[key]["url"], "cited_by_pages": lost[key]["cited_by_pages"],
                 "archive": "Common Crawl", "collection": best["collection"],
                 "captured": best["timestamp"], "warc": best["filename"],
                 "offset": int(best["offset"]), "length": int(best["length"])}
        if args.recovered_dir:
            try:
                entry.update(fetch_warc(best, args.recovered_dir))
            except Exception as e:
                entry["fetch_error"] = str(e)
        recovered.append(entry)

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domain": domain,
        "studied_period_ends": to_ts,
        "articles_marked_lost": len(lost),
        "wayback_recheck": {
            "method": "full domain capture index pulled and compared offline, so "
                      "every lost URL is re-tested rather than a sample",
            "distinct_urls_with_200_in_period": len(ok200),
            "distinct_urls_captured_ever": len(anycap),
            "lost_but_actually_have_200": len(has_200),
            "lost_captured_but_never_200": len(no_200),
            "lost_with_no_capture_at_all": len(absent),
            "false_negatives": [lost[k]["url"] for k in has_200],
            "captured_but_empty": [lost[k]["url"] for k in no_200],
        },
        "common_crawl": {
            "collections_queried": len(cc_asked),
            "collections_with_records": sum(1 for c in cc_asked if c["records"]),
            "records": len(cc_hits),
            "distinct_urls": len({norm(r["url"]) for r in cc_hits}),
            "distinct_urls_with_200": len(cc_200),
            "already_held": sorted(set(cc_200) & held),
            "per_collection": cc_asked,
        },
        "recovered_elsewhere": recovered,
        "not_reachable": UNREACHABLE,
    }

    out = args.out or os.path.join(os.path.dirname(args.db), "cross-check.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    w = report["wayback_recheck"]
    print(f"\nlost articles re-tested:  {len(lost)}")
    print(f"  false negatives:        {w['lost_but_actually_have_200']}")
    print(f"  captured but empty:     {w['lost_captured_but_never_200']}")
    print(f"  no record at all:       {w['lost_with_no_capture_at_all']}")
    print(f"recovered from elsewhere: {len(recovered)}")
    for r in recovered:
        print(f"  {r['archive']} {r['captured'][:8]}  {r['url']}"
              f"  (cited by {r['cited_by_pages']} pages)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
