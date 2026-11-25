"""URL patterns a project keeps out of its corpus.

Crawlers capture a site's plumbing along with its content: CMS admin and login
pages, feeds, stylesheets, robots.txt. Those are not what is being studied,
and analyzing them costs model calls and pollutes the findings.

Patterns are globs matched case-insensitively against the URL's path, e.g.
``/wp-admin/`` or ``*.css``. ``*`` matches anything, including ``/``. A
pattern ending in ``/`` covers everything under that folder. A pattern
containing ``?`` is matched against path and query together. Lines starting
with ``#`` are comments.

Exclusion never deletes: a matched capture leaves the analyzed scope, and one
not yet downloaded is not fetched. Removing the pattern brings it back.
"""
from fnmatch import fnmatchcase
from typing import List, Optional
from urllib.parse import unquote, urlsplit

EXCLUDED_PREFIX = "Excluded by pattern"

# Plumbing of the CMSs common in old sites. Uploaded media (wp-content/uploads)
# is content and is deliberately not listed.
SUGGESTED = """\
/wp-admin/
/wp-includes/
/wp-json/
/wp-content/themes/
/wp-content/plugins/
/wp-login.php
/wp-cron.php
/xmlrpc.php
*/feed/
*/trackback/
*/embed/
/cgi-bin/
/cdn-cgi/
/robots.txt
/favicon.ico
*.css
*.js
"""


def parse(text: Optional[str]) -> List[str]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _glob(pattern: str) -> str:
    if pattern.endswith("/"):
        pattern += "*"
    if not pattern.startswith(("/", "*")):
        pattern = "*" + pattern
    return pattern


def match(url: str, patterns: List[str]) -> Optional[str]:
    """The first pattern that excludes ``url``, or None."""
    if not patterns:
        return None
    parts = urlsplit(url)
    path = unquote(parts.path or "/").lower()
    full = path + ("?" + unquote(parts.query).lower() if parts.query else "")
    for p in patterns:
        target = full if "?" in p else path
        g = _glob(p)
        # "/feed/" style patterns also cover the folder itself without a slash
        if fnmatchcase(target, g) or (p.endswith("/") and fnmatchcase(target + "/", g)):
            return p
    return None


def reason(pattern: str) -> str:
    return f"{EXCLUDED_PREFIX}: {pattern}"
