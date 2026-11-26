"""Domain enrichment: free RDAP/WHOIS lookup and Wayback timeline aggregation.

Ownership uses RDAP (Registration Data Access Protocol) via the public rdap.org
bootstrap — free, structured, no API key. For old domains the registrant is usually
redacted/GDPR-masked; we still surface registrar, registration/expiry dates,
nameservers and status. The "who actually ran it" signal comes mostly from people
mentioned in the archived content (aggregated separately by the LLM/analysis layer).
"""
import re

import httpx
from collections import Counter
from typing import Dict, List, Any, Optional

RDAP_BASE = "https://rdap.org/domain"

# Registries that run RDAP but are missing from IANA's bootstrap, which is all
# rdap.org consults. For these rdap.org answers "no RDAP service", so they are
# asked directly.
_DIRECT_RDAP = {"ch": "https://rdap.nic.ch/domain", "li": "https://rdap.nic.ch/domain"}

# Second-level labels that are public suffixes under a country code
# (example.co.uk, example.com.au, example.gob.mx). A lookup never stops at one
# of these: "co.uk" is not anyone's registration.
_GENERIC_SLDS = {"co", "com", "net", "org", "gov", "gob", "edu", "ac", "mil",
                 "ne", "or", "go", "nic", "ltd", "plc", "sch", "nom", "info", "biz"}


def host_of(domain: str) -> str:
    """Bare hostname from whatever the project was created with."""
    d = domain.strip().lower()
    d = re.sub(r"^[a-z][a-z0-9+.-]*://", "", d)
    d = d.split("/", 1)[0].split("?", 1)[0].split("@")[-1].split(":", 1)[0]
    return d.strip(".")


def registration_candidates(domain: str) -> List[str]:
    """The host and each parent down to two labels, most specific first.

    Registries only hold the domain that was registered (cern.ch), not the
    hosts under it (info.cern.ch), so a lookup walks up until one answers.
    """
    labels = [x for x in host_of(domain).split(".") if x]
    out = []
    for i in range(len(labels) - 1):
        cand = labels[i:]
        if len(cand) == 2 and len(cand[1]) == 2 and cand[0] in _GENERIC_SLDS:
            break
        out.append(".".join(cand))
    return out or [host_of(domain)]


def _vcard_to_dict(entity: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extract name/org/email from an RDAP entity's jCard (vcardArray)."""
    info: Dict[str, Optional[str]] = {}
    vcard = entity.get("vcardArray")
    if not (isinstance(vcard, list) and len(vcard) == 2):
        return info
    for field in vcard[1]:
        if not (isinstance(field, list) and len(field) >= 4):
            continue
        key = field[0]
        value = field[3]
        if key == "fn":
            info["name"] = value
        elif key == "org":
            info["org"] = value if isinstance(value, str) else (value[-1] if value else None)
        elif key == "email":
            info["email"] = value
    return info


def _find_entity(entities: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for entity in entities or []:
        roles = entity.get("roles") or []
        if role in roles:
            return entity
    return None


async def fetch_rdap(domain: str) -> Dict[str, Any]:
    """Look up current registration data via RDAP. Returns a normalized dict.

    `whois_source` is one of: rdap, unregistered, unavailable (the TLD's
    registry offers no RDAP), error.

    "unregistered" is a finding, not a failure. For a dead site it is often the
    most telling field on the card: b1mas4.com returns 404 from the authoritative
    .com registry and resolves to nothing, meaning the domain lapsed again after
    its squatters were done with it. Reporting that as an unavailable lookup
    would hide it behind what looks like a broken network call.
    """
    # rdap.org redirects to the authoritative registry/registrar RDAP server.
    candidates = registration_candidates(domain)
    result: Dict[str, Any] = {
        "registrar": None,
        "whois_created": None,
        "whois_expires": None,
        "whois_updated": None,
        "nameservers": [],
        "whois_status": [],
        "registrant": None,
        "whois_raw": None,
        "whois_source": "unavailable",
        # The domain the result is about: the registered domain, which for a
        # site on a subdomain is a parent of the project's domain.
        "whois_domain": candidates[-1],
    }

    data = None
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            base = _DIRECT_RDAP.get(candidates[-1].rsplit(".", 1)[-1], RDAP_BASE)
            for cand in candidates:
                response = await client.get(f"{base}/{cand}",
                                            headers={"Accept": "application/rdap+json"})
                if response.status_code == 404:
                    # rdap.org also says 404 when the TLD has no RDAP server at
                    # all. That is "unknown", not "unregistered".
                    if "no rdap service" in response.text[:500].lower():
                        return result
                    continue
                if response.status_code in (400, 422) and cand != candidates[-1]:
                    # Some registries (SWITCH for .ch) reject a name below the
                    # registered domain as malformed instead of answering 404.
                    continue
                response.raise_for_status()
                data = response.json()
                result["whois_domain"] = cand
                break
    except (httpx.HTTPError, ValueError):
        result["whois_source"] = "error"
        return result
    if data is None:
        # The registry is authoritative: no record for the host or any of its
        # parents means the registered domain is not currently registered.
        result["whois_source"] = "unregistered"
        return result

    result["whois_source"] = "rdap"
    result["whois_raw"] = data

    # Events -> dates
    for event in data.get("events", []) or []:
        action = event.get("eventAction")
        date = event.get("eventDate")
        if action == "registration":
            result["whois_created"] = date
        elif action == "expiration":
            result["whois_expires"] = date
        elif action in ("last changed", "last update of RDAP database"):
            result["whois_updated"] = result["whois_updated"] or date

    # Nameservers & status
    result["nameservers"] = [
        ns.get("ldhName") for ns in data.get("nameservers", []) or [] if ns.get("ldhName")
    ]
    result["whois_status"] = data.get("status", []) or []

    # Registrar & registrant from entities
    entities = data.get("entities", []) or []
    registrar = _find_entity(entities, "registrar")
    if registrar:
        reg = _vcard_to_dict(registrar)
        result["registrar"] = reg.get("name") or reg.get("org")

    registrant = _find_entity(entities, "registrant")
    if registrant:
        info = _vcard_to_dict(registrant)
        if info:
            result["registrant"] = info

    return result


def build_timeline(timestamps: List[str]) -> Dict[str, Any]:
    """Aggregate a Wayback capture timeline from archived-file timestamps.

    timestamps are Wayback YYYYMMDDHHMMSS strings.
    """
    valid = sorted(t for t in timestamps if t and len(t) >= 8 and t[:8].isdigit())
    if not valid:
        return {
            "first_seen": None,
            "last_seen": None,
            "per_year": {},
            "went_dark": None,
            "total_captures": 0,
        }

    def fmt(ts: str) -> str:
        return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"

    per_year = Counter(t[:4] for t in valid)
    last_year = valid[-1][:4]

    return {
        "first_seen": fmt(valid[0]),
        "last_seen": fmt(valid[-1]),
        "per_year": dict(sorted(per_year.items())),
        "went_dark": last_year,
        "total_captures": len(valid),
    }


def person_to_str(person: Any) -> str:
    """Normalize a 'person' entry to a name string.

    Different LLM models/runs store `people_mentioned` either as plain strings or
    as objects like {"name": "..."} / {"person": "..."}; handle both gracefully.
    """
    if person is None:
        return ""
    if isinstance(person, dict):
        for key in ("name", "fn", "person", "full_name", "username"):
            value = person.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in person.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    return str(person).strip()


def aggregate_people(analyses: List[Any], canon: Any = None, limit: int = 25) -> List[Dict[str, Any]]:
    """Aggregate people mentioned across file analyses, ranked by mention count.

    ``canon`` (services/people.Canon) merges spellings of one person."""
    counter: Counter = Counter()
    for analysis in analyses:
        for person in (getattr(analysis, "people_mentioned", None) or []):
            name = person_to_str(person)
            if name:
                counter[canon.canonical(name) if canon else name] += 1
    return [{"name": canon.display(k) if canon else k, "mentions": count}
            for k, count in counter.most_common(limit)]
