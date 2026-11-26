"""The sites a site linked to, as candidates for their own projects.

A site's outbound links are its neighbourhood: the festival it covered, the
studio it worked with, the blogs it read. Many of those sites are gone too.
This groups the recovered link targets by site and lets a person start a
project on one, carrying the provenance ("started from the links of X").

Nothing is searched, downloaded or analyzed until that new project's own
workflow is run.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .enrichment import host_of, registration_candidates

# Platforms whose pages are someone's account on a shared service, not a site
# with a history of its own. Shown, but not offered as a project.
PLATFORMS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com", "youtu.be",
    "flickr.com", "linkedin.com", "pinterest.com", "vimeo.com", "soundcloud.com",
    "google.com", "goo.gl", "bit.ly", "t.co", "ow.ly", "tinyurl.com", "eepurl.com",
    "list-manage.com", "list-manage1.com", "list-manage2.com", "mailchimp.com",
    "wikipedia.org", "wikimedia.org", "about.me", "gravatar.com", "addthis.com",
    "feedburner.com", "creativecommons.org", "wordpress.org", "w3.org", "apple.com",
    "microsoft.com", "adobe.com", "amazon.com", "paypal.com", "github.com", "archive.org",
    "whatsapp.com", "tiktok.com", "reddit.com", "imgur.com", "issuu.com", "scribd.com",
}
# Hosting services where each subdomain is a separate site.
HOSTS = {
    "blogspot.com", "wordpress.com", "tumblr.com", "geocities.com", "tripod.com",
    "angelfire.com", "livejournal.com", "github.io", "neocities.org", "typepad.com",
    "wix.com", "weebly.com", "over-blog.com", "blogia.com", "jimdo.com",
}


def site_of(domain: str) -> str:
    """The site a link belongs to: its registered domain, or the subdomain on a
    hosting service (somebody.blogspot.com)."""
    host = host_of(domain or "")
    if host.startswith("www."):
        host = host[4:]
    registered = registration_candidates(host)[-1] if host else ""
    if registered in HOSTS and host != registered:
        return host
    return registered


def is_platform(site: str) -> bool:
    return site in PLATFORMS


async def summary(db, project_id: int, limit: int = 60) -> List[Dict[str, Any]]:
    rows = (await db.execute(text("""
        SELECT lt.domain, COUNT(*) AS targets,
               SUM(lt.recovery_status = 'recovered') AS recovered,
               MIN(lt.first_cited_ts), MAX(lt.last_cited_ts)
        FROM link_targets lt
        WHERE lt.project_id = :pid AND lt.is_internal = 0 AND lt.domain IS NOT NULL
        GROUP BY lt.domain
    """), {"pid": project_id})).fetchall()
    citing = (await db.execute(text("""
        SELECT lt.domain, af.original_url
        FROM page_links pl JOIN link_targets lt ON lt.id = pl.target_id
        JOIN archived_files af ON af.id = pl.source_file_id
        WHERE pl.project_id = :pid AND lt.is_internal = 0
        GROUP BY lt.domain, af.original_url
    """), {"pid": project_id})).fetchall()
    sites: Dict[str, Dict[str, Any]] = {}
    for domain, targets, recovered, first, last in rows:
        site = site_of(domain)
        if not site:
            continue
        s = sites.setdefault(site, {"site": site, "targets": 0, "recovered": 0, "pages": set(),
                                    "first": first, "last": last, "platform": is_platform(site)})
        s["targets"] += int(targets or 0)
        s["recovered"] += int(recovered or 0)
        s["first"] = min(filter(None, [s["first"], first]), default=None)
        s["last"] = max(filter(None, [s["last"], last]), default=None)
    for domain, url in citing:
        site = site_of(domain)
        if site in sites:
            sites[site]["pages"].add(url)
    own = site_of((await db.execute(text("SELECT domain FROM projects WHERE id = :pid"),
                                    {"pid": project_id})).scalar() or "")
    existing = await projects_by_site(db)
    out = []
    for s in sites.values():
        if s["site"] == own:
            continue
        s["pages"] = len(s["pages"])
        s["project_id"] = existing.get(s["site"])
        out.append(s)
    # Sites first, platforms after; within each, the most-cited first.
    out.sort(key=lambda s: (s["platform"], -s["pages"], -s["recovered"], s["site"]))
    return out[:limit]


async def projects_by_site(db) -> Dict[str, int]:
    rows = (await db.execute(text("SELECT id, domain FROM projects ORDER BY id"))).fetchall()
    out: Dict[str, int] = {}
    for pid, domain in rows:
        out.setdefault(site_of(domain), pid)
    return out


async def existing_project(db, site: str) -> Optional[int]:
    return (await projects_by_site(db)).get(site_of(site))
