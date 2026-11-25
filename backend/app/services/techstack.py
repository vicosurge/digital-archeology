"""Offline technology-stack detection over archived HTML.

Uses the vendored Wappalyzer fingerprint database (backend/app/data/fingerprints/)
to identify CMS, frameworks, JS libraries, analytics, servers, etc. from the HTML
that was already downloaded from the Wayback Machine.

Note: Wayback's replay download stores the response *body* only, not the original
response headers, so header/cookie based fingerprints cannot fire. Detection relies
on `html`, `meta` and `scriptSrc` patterns, which still reliably catch the things
that matter for old sites (phpBB, vBulletin, WordPress, jQuery, generator tags,
Google Analytics, FrontPage, etc.).
"""
import os
import re
import json
import glob
from typing import Dict, List, Any, Optional

FINGERPRINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "fingerprints"
)

# How many downloaded HTML files to sample per project for detection.
MAX_FILES_SAMPLED = 40
# Cap bytes read per file (fingerprints only need head/markup signals).
MAX_BYTES_PER_FILE = 400_000

_META_RE = re.compile(
    r"""<meta[^>]+?(?:name|property|http-equiv)\s*=\s*["']?([^"'>\s]+)["']?"""
    r"""[^>]+?content\s*=\s*["']([^"']*)["']""",
    re.I,
)
_SCRIPT_SRC_RE = re.compile(r"""<script[^>]+src\s*=\s*["']([^"']+)["']""", re.I)


class CompiledPattern:
    """A single Wappalyzer pattern: regex plus optional version template."""

    __slots__ = ("regex", "version")

    def __init__(self, regex: "re.Pattern", version: Optional[str]):
        self.regex = regex
        self.version = version


def _parse_pattern(raw: str) -> Optional[CompiledPattern]:
    """Parse the `regex\\;version:\\1\\;confidence:50` Wappalyzer pattern syntax."""
    if not isinstance(raw, str):
        return None
    parts = raw.split("\\;")
    expr = parts[0]
    version = None
    for tag in parts[1:]:
        if tag.startswith("version:"):
            version = tag[len("version:"):]
    try:
        return CompiledPattern(re.compile(expr, re.I), version)
    except re.error:
        # Some upstream patterns use PCRE features Python's re rejects; skip them.
        return None


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


class TechStackDetector:
    """Loads fingerprints once and matches them against archived HTML."""

    def __init__(self):
        self._loaded = False
        self.categories: Dict[str, str] = {}
        # name -> {"cats": [..], "implies": [..],
        #          "html": [CompiledPattern], "scriptSrc": [...],
        #          "meta": {metaname: [CompiledPattern]}}
        self.tech: Dict[str, Dict[str, Any]] = {}

    def _load(self):
        if self._loaded:
            return

        cats_path = os.path.join(FINGERPRINT_DIR, "categories.json")
        if os.path.exists(cats_path):
            try:
                with open(cats_path, encoding="utf-8") as f:
                    raw_cats = json.load(f)
                for cid, val in raw_cats.items():
                    name = val.get("name") if isinstance(val, dict) else str(val)
                    self.categories[str(cid)] = name or str(cid)
            except (json.JSONDecodeError, OSError):
                pass

        for fp in glob.glob(os.path.join(FINGERPRINT_DIR, "tech_*.json")):
            try:
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for name, spec in data.items():
                if not isinstance(spec, dict):
                    continue
                html_pats = [p for p in (_parse_pattern(x) for x in _as_list(spec.get("html"))) if p]
                src_pats = [p for p in (_parse_pattern(x) for x in _as_list(spec.get("scriptSrc"))) if p]
                meta_pats: Dict[str, List[CompiledPattern]] = {}
                meta = spec.get("meta")
                if isinstance(meta, dict):
                    for mname, mval in meta.items():
                        pats = [p for p in (_parse_pattern(x) for x in _as_list(mval)) if p]
                        if pats:
                            meta_pats[mname.lower()] = pats
                # Only keep technologies we can actually detect from HTML.
                if not (html_pats or src_pats or meta_pats):
                    continue
                self.tech[name] = {
                    "cats": [str(c) for c in spec.get("cats", [])],
                    "implies": _as_list(spec.get("implies")),
                    "html": html_pats,
                    "scriptSrc": src_pats,
                    "meta": meta_pats,
                }

        self._loaded = True

    def _category_names(self, cat_ids: List[str]) -> List[str]:
        return [self.categories.get(c, c) for c in cat_ids]

    @staticmethod
    def _extract_version(pat: CompiledPattern, match: "re.Match") -> Optional[str]:
        if not pat.version:
            return None
        version = pat.version
        # Resolve simple \1..\9 backreferences from the match groups.
        for i in range(1, (match.re.groups or 0) + 1):
            token = "\\" + str(i)
            if token in version:
                group = match.group(i) or ""
                version = version.replace(token, group)
        version = version.strip()
        return version or None

    def detect_in_html(self, html: str, acc: Dict[str, Dict[str, Any]]):
        """Run all fingerprints against one HTML document, merging into `acc`."""
        meta_map = {name.lower(): content for name, content in _META_RE.findall(html)}
        script_srcs = _SCRIPT_SRC_RE.findall(html)

        for name, spec in self.tech.items():
            version = None
            evidence = None

            for pat in spec["html"]:
                m = pat.regex.search(html)
                if m:
                    evidence = "html"
                    version = version or self._extract_version(pat, m)
                    break

            if not evidence:
                for pat in spec["scriptSrc"]:
                    for src in script_srcs:
                        m = pat.regex.search(src)
                        if m:
                            evidence = "scriptSrc"
                            version = version or self._extract_version(pat, m)
                            break
                    if evidence:
                        break

            if not evidence and spec["meta"]:
                for mname, pats in spec["meta"].items():
                    content = meta_map.get(mname)
                    if content is None:
                        continue
                    for pat in pats:
                        m = pat.regex.search(content)
                        if m:
                            evidence = "meta"
                            version = version or self._extract_version(pat, m)
                            break
                    if evidence:
                        break

            if not evidence:
                continue

            entry = acc.setdefault(
                name,
                {
                    "name": name,
                    "categories": self._category_names(spec["cats"]),
                    "version": None,
                    "evidence": set(),
                    "implies": spec["implies"],
                    "_implied": False,
                },
            )
            entry["evidence"].add(evidence)
            if version and not entry["version"]:
                entry["version"] = version

    def _resolve_implies(self, acc: Dict[str, Dict[str, Any]]):
        """Add technologies implied by detected ones (e.g. phpBB implies PHP)."""
        queue = list(acc.keys())
        while queue:
            name = queue.pop()
            for implied in acc.get(name, {}).get("implies", []):
                # Implies entries can carry confidence tags too.
                implied_name = implied.split("\\;")[0]
                if implied_name in acc or implied_name not in self.tech:
                    continue
                spec = self.tech[implied_name]
                acc[implied_name] = {
                    "name": implied_name,
                    "categories": self._category_names(spec["cats"]),
                    "version": None,
                    "evidence": set(),
                    "implies": spec["implies"],
                    "_implied": True,
                }
                queue.append(implied_name)

    def detect_in_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Detect technologies across a set of local HTML files; union the results."""
        self._load()
        acc: Dict[str, Dict[str, Any]] = {}

        for path in file_paths[:MAX_FILES_SAMPLED]:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read(MAX_BYTES_PER_FILE)
            except OSError:
                continue
            self.detect_in_html(html, acc)

        self._resolve_implies(acc)

        results = []
        for entry in acc.values():
            results.append({
                "name": entry["name"],
                "categories": entry["categories"],
                "version": entry["version"],
                "evidence": "implied" if entry["_implied"] else ", ".join(sorted(entry["evidence"])),
            })
        # Sort: directly detected first, then by name.
        results.sort(key=lambda e: (e["evidence"] == "implied", e["name"].lower()))
        return results


# Global instance (fingerprints are loaded lazily on first use).
techstack_detector = TechStackDetector()
