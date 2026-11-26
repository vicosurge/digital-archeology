"""One person, one entry: merging the spellings of a name.

The model writes whatever the page wrote: "Tim Berners-Lee", "Tim
Berners Lee", "Berners-Lee", "Tim BL". Counted raw, one person becomes four.

Three layers, applied in this order:

1. **Folding** is always on: case, accents, dots and hyphens do not make a new
   person ("Jean-François Groff" = "jean francois groff").
2. **The automatic rule** is ``apparatus.merge_variants``, unchanged: a name
   merges into a longer one only when all its words are in it and exactly one
   longer candidate matches. "Berners-Lee" joins "Tim Berners-Lee"; a bare
   "López" with two Lópezes in the corpus stays alone.
3. **The historian's decisions**, stored per project in ``person_aliases``:
   "these are the same person" (kind ``merge``) and "these are not"
   (kind ``separate``, which also overrides the automatic rule). Merges
   follow chains, so merging into a name that was itself merged works.

Nicknames and initials ("Tim BL", "T. Berners-Lee") are never merged
automatically. They are offered as suggestions (``suggestions``) for a person
to accept or reject.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy import text

from .apparatus import fold, merge_variants
from .enrichment import person_to_str

MIN_NAME_LENGTH = 2
AMBIGUOUS_MAX = 4

PERSON_VALUE = (
    "CASE WHEN json_valid(je.value) AND json_type(je.value) = 'object' "
    "THEN json_extract(je.value, '$.name') ELSE je.value END"
)


def key(name: str) -> str:
    """The folded key a spelling is counted under."""
    return " ".join(fold(re.sub(r"[-‐‑–_]", " ", name or "")).split())


@dataclass
class Canon:
    """Resolves any spelling to its person and that person's display name."""
    to_canonical: Dict[str, str] = field(default_factory=dict)   # key -> canonical key
    spellings: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    manual: Set[str] = field(default_factory=set)                # keys merged by hand

    def canonical(self, name: Any) -> str:
        k = key(person_to_str(name))
        return self.to_canonical.get(k, k)

    def display(self, canonical_key: str) -> str:
        """The most common spelling among everything merged into this person."""
        counts = self.spellings.get(canonical_key)
        if counts:
            return counts.most_common(1)[0][0]
        return canonical_key

    def name(self, raw: Any) -> str:
        return self.display(self.canonical(raw))

    def variants(self, canonical_key: str) -> List[str]:
        """Every raw spelling that resolves to this person."""
        return sorted(self.spellings.get(canonical_key, {}).keys(), key=str.lower)

    def keys_of(self, canonical_key: str) -> List[str]:
        return sorted(k for k, c in self.to_canonical.items() if c == canonical_key)


def resolve(names: Counter, merges: Dict[str, str], separate: Set[Tuple[str, str]]) -> Canon:
    """Build the canonicalizer from raw spellings and their counts.

    ``merges`` maps an alias key to the key it was merged into; ``separate``
    holds unordered key pairs a person said are different people.
    """
    keys: Counter = Counter()
    for raw, n in names.items():
        k = key(raw)
        if len(k) >= MIN_NAME_LENGTH:
            keys[k] += n
    auto = merge_variants(list(keys))

    def blocked(a: str, b: str) -> bool:
        return (a, b) in separate or (b, a) in separate

    # Manual merges first, then the automatic rule for what they leave alone.
    target: Dict[str, str] = {}
    for k in keys:
        if k in merges:
            target[k] = merges[k]
        elif auto.get(k, k) != k and not blocked(k, auto[k]):
            target[k] = auto[k]
        else:
            target[k] = k
    for alias, into in merges.items():          # a merge may name a key no page uses now
        target.setdefault(alias, into)
        target.setdefault(into, merges.get(into, into))

    def follow(k: str) -> str:
        seen = {k}
        while target.get(k, k) != k and target[k] not in seen:
            k = target[k]
            seen.add(k)
        return k

    canon = Canon(manual=set(merges))
    for k in target:
        canon.to_canonical[k] = follow(k)
    for raw, n in names.items():
        k = key(raw)
        if len(k) >= MIN_NAME_LENGTH:
            canon.spellings[canon.to_canonical.get(k, k)][raw.strip()] += n
    return canon


async def raw_names(db, project_id: int, in_scope_only: bool = False) -> Counter:
    """Every spelling the analyses used, with how many pages used it."""
    scope = "AND af.in_scope = 1" if in_scope_only else ""
    rows = (await db.execute(text(f"""
        SELECT {PERSON_VALUE} AS person, COUNT(DISTINCT af.id)
        FROM archived_files af JOIN file_analyses fa ON fa.file_id = af.id,
             json_each(fa.people_mentioned) je
        WHERE af.project_id = :pid AND af.status = 'PROCESSED' AND fa.source_file_id IS NULL {scope}
          AND person IS NOT NULL AND trim(person) != ''
        GROUP BY person
    """), {"pid": project_id})).fetchall()
    out: Counter = Counter()
    for name, n in rows:
        out[str(name).strip()] += int(n)
    return out


async def decisions(db, project_id: int) -> Tuple[Dict[str, str], Set[Tuple[str, str]]]:
    rows = (await db.execute(text(
        "SELECT alias_key, canonical_key, kind FROM person_aliases WHERE project_id = :pid"),
        {"pid": project_id})).fetchall()
    merges = {a: c for a, c, kind in rows if kind == "merge"}
    separate = {(a, c) for a, c, kind in rows if kind == "separate"}
    return merges, separate


async def canonicalizer(db, project_id: int, extra: Iterable[Any] = ()) -> Canon:
    """The project's canonicalizer. ``extra`` adds spellings from elsewhere
    (close readings) so they resolve too."""
    names = await raw_names(db, project_id)
    for raw in extra:
        s = person_to_str(raw)
        if s:
            names[s] += 0 if s in names else 1
    merges, separate = await decisions(db, project_id)
    return resolve(names, merges, separate)


# ---------- the historian's decisions ----------

async def merge(db, project_id: int, alias_keys: Iterable[str], into_key: str) -> int:
    """Record that every alias is the same person as ``into_key``."""
    n = 0
    for alias in {a for a in alias_keys if a and a != into_key}:
        await db.execute(text(
            "DELETE FROM person_aliases WHERE project_id = :pid AND "
            "((alias_key = :a AND kind = 'merge') OR (kind = 'separate' AND "
            "((alias_key = :a AND canonical_key = :c) OR (alias_key = :c AND canonical_key = :a))))"),
            {"pid": project_id, "a": alias, "c": into_key})
        await db.execute(text(
            "INSERT INTO person_aliases (project_id, alias_key, canonical_key, kind) "
            "VALUES (:pid, :a, :c, 'merge')"), {"pid": project_id, "a": alias, "c": into_key})
        n += 1
    # Anything that pointed at an alias now points at the new person directly.
    await db.execute(text(
        "UPDATE person_aliases SET canonical_key = :c WHERE project_id = :pid AND kind = 'merge' "
        "AND canonical_key IN (SELECT alias_key FROM person_aliases WHERE project_id = :pid "
        "AND kind = 'merge' AND canonical_key = :c) AND alias_key != :c"),
        {"pid": project_id, "c": into_key})
    return n


async def keep_separate(db, project_id: int, a: str, b: str) -> None:
    await db.execute(text(
        "DELETE FROM person_aliases WHERE project_id = :pid AND kind = 'separate' AND "
        "((alias_key = :a AND canonical_key = :b) OR (alias_key = :b AND canonical_key = :a))"),
        {"pid": project_id, "a": a, "b": b})
    await db.execute(text(
        "INSERT INTO person_aliases (project_id, alias_key, canonical_key, kind) "
        "VALUES (:pid, :a, :b, 'separate')"), {"pid": project_id, "a": a, "b": b})


async def unmerge(db, project_id: int, canonical_key: str) -> int:
    """Undo every hand merge into this person (the automatic rule still applies)."""
    result = await db.execute(text(
        "DELETE FROM person_aliases WHERE project_id = :pid AND kind = 'merge' "
        "AND (canonical_key = :c OR alias_key = :c)"), {"pid": project_id, "c": canonical_key})
    return result.rowcount or 0


async def forget_separate(db, project_id: int, a: str, b: str) -> None:
    await db.execute(text(
        "DELETE FROM person_aliases WHERE project_id = :pid AND kind = 'separate' AND "
        "((alias_key = :a AND canonical_key = :b) OR (alias_key = :b AND canonical_key = :a))"),
        {"pid": project_id, "a": a, "b": b})


# ---------- suggestions ----------

def _initials_match(short: List[str], full: List[str]) -> bool:
    """'bl' or 'b l' against ['berners', 'lee']."""
    if not short or not full:
        return False
    joined = "".join(short)
    if not joined.isalpha() or len(joined) > 4 or any(len(t) > 3 for t in short):
        return False
    return joined == "".join(t[0] for t in full)


def is_initials_variant(a: str, b: str) -> bool:
    """'tim bl' ~ 'tim berners lee', 't berners lee' ~ 'tim berners lee'."""
    ta, tb = a.split(), b.split()
    if len(ta) < 2 or len(tb) < 2 or a == b:
        return False
    for x, y in ((ta, tb), (tb, ta)):
        # Same first name, the rest abbreviated.
        if x[0] == y[0] and len(y) > len(x) - 1 and _initials_match(x[1:], y[1:]):
            return True
        # First name as an initial, same surname(s).
        if len(x[0]) == 1 and len(y[0]) > 1 and x[0] == y[0][0] and x[1:] == y[1:]:
            return True
    return False


def suggestions(canon: Canon, separate: Set[Tuple[str, str]], limit: int = 60) -> List[Dict[str, Any]]:
    """Pairs that may be one person, for a person to decide.

    Two rules, both of which the automatic merge refuses on purpose:
    initials ("Tim BL", "T. Berners-Lee") and a short name that fits more than
    one fuller name ("López" with "Ana López" and "Luis López"; listed once
    per candidate).
    """
    people = {c for c in canon.to_canonical.values() if canon.spellings.get(c)}
    weight = {c: sum(canon.spellings[c].values()) for c in people}
    done = set(separate) | {(b, a) for a, b in separate}
    out: List[Dict[str, Any]] = []

    def add(a: str, b: str, reason: str) -> None:
        """Suggest merging a into b."""
        if (a, b) in done or a == b:
            return
        done.add((a, b)); done.add((b, a))
        out.append({"alias": a, "into": b, "alias_name": canon.display(a),
                    "into_name": canon.display(b), "alias_pages": weight.get(a, 0),
                    "into_pages": weight.get(b, 0), "reason": reason})

    # Initials: only names sharing a first word, or sharing everything after
    # it, can match, so compare within those groups instead of every pair.
    by_first: Dict[str, List[str]] = defaultdict(list)
    by_rest: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for k in people:
        t = k.split()
        if len(t) >= 2:
            by_first[t[0]].append(k)
            by_rest[tuple(t[1:])].append(k)
    for group in list(by_first.values()) + list(by_rest.values()):
        for a, b in combinations(sorted(group), 2):
            if is_initials_variant(a, b):
                # The abbreviated form merges into the spelled-out one.
                a, b = (a, b) if len(a) < len(b) else (b, a)
                add(a, b, "initials")

    # A shorter name inside a few fuller ones. Past AMBIGUOUS_MAX it is a common
    # first name ("Luis" in 27 names), not a person to place.
    tokens = {k: frozenset(k.split()) for k in people}
    postings: Dict[str, Set[str]] = defaultdict(set)
    for k, t in tokens.items():
        for w in t:
            postings[w].add(k)
    for short in sorted(people):
        st = tokens[short]
        if not st:
            continue
        lists = sorted((postings[w] for w in st), key=len)
        fuller = sorted(c for c in set(lists[0]).intersection(*lists[1:]) if st < tokens[c])
        if 2 <= len(fuller) <= AMBIGUOUS_MAX:
            for c in fuller:
                add(short, c, f"fits {len(fuller)} names")
    out.sort(key=lambda s: -(s["alias_pages"] + s["into_pages"]))
    return out[:limit]


def groups(canon: Canon) -> List[Dict[str, Any]]:
    """Every person with the spellings merged into them, most-cited first."""
    out = []
    for c, counts in canon.spellings.items():
        if not counts:
            continue
        out.append({"key": c, "name": canon.display(c), "pages": sum(counts.values()),
                    "variants": sorted(counts.items(), key=lambda kv: -kv[1]),
                    "manual": any(canon.to_canonical.get(k) == c and k in canon.manual
                                  for k in canon.to_canonical)})
    out.sort(key=lambda g: (-g["pages"], g["name"].lower()))
    return out
