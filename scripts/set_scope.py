#!/usr/bin/env python3
"""Set (or clear) a project's analysis scope window.

A domain's capture history is not necessarily one site. When a domain expires
and is re-registered, later captures share the URL and nothing else. This script
records which era of a project's captures is the subject of study; everything
outside the window is kept in the database but excluded from analysis, from the
results aggregates and from the archival card.

Nothing is deleted and no file's status or analysis is touched -- the flag is
recomputed from the window, so it can be widened, narrowed or cleared later and
the excluded captures are still there. Run with no arguments to see the current
scope and a per-year capture breakdown before deciding.

    # inspect
    python3 scripts/set_scope.py --project 4

    # set: b1mas4 was a Tijuana news site until it expired; the domain was then
    # taken over by a pharmacy-spam farm (2019) and a gambling affiliate (2022)
    python3 scripts/set_scope.py --project 4 --end 20170101000000 \\
        --reason "b1mas4.com was a Tijuana news site through 2016. The domain
                  then expired and was taken over -- pharmacy spam from 2019,
                  a gambling affiliate from 2022. Only the original site is
                  analyzed; later captures are kept as evidence of the takeover."

    # clear
    python3 scripts/set_scope.py --project 4 --clear
"""
import argparse
import asyncio
import os
import sys
import textwrap

# The `app` package lives under backend/ in a source checkout and at /app inside
# the container; this script runs in both.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.join(_HERE, "..", "backend"), "/app", _HERE):
    if os.path.isfile(os.path.join(_candidate, "app", "models.py")):
        sys.path.insert(0, _candidate)
        break

from sqlalchemy import select, text  # noqa: E402

from app.database import async_session_maker, init_db  # noqa: E402
from app.models import Project  # noqa: E402
from app.routers.projects import apply_scope  # noqa: E402


async def show(session, project: Project) -> None:
    print(f"project {project.id}: {project.name}  ({project.domain})")
    print(f"  scope_start : {project.scope_start or '(unbounded)'}")
    print(f"  scope_end   : {project.scope_end or '(unbounded)'}")
    print(f"  reason      : {project.scope_reason or '(none)'}")

    rows = (await session.execute(
        text("SELECT substr(timestamp, 1, 4) AS year, COUNT(*), SUM(in_scope) "
             "FROM archived_files WHERE project_id = :pid "
             "GROUP BY year ORDER BY year"),
        {"pid": project.id},
    )).fetchall()

    print("\n  year   captures   in scope")
    for year, total, in_scope in rows:
        marker = "" if int(in_scope or 0) == int(total) else "   <- partly excluded" \
            if in_scope else "   <- excluded"
        print(f"  {year or '????'}   {total:>8}   {int(in_scope or 0):>8}{marker}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=int, required=True, help="project id")
    ap.add_argument("--start", help="inclusive lower bound, Wayback YYYYMMDDHHMMSS")
    ap.add_argument("--end", help="exclusive upper bound, Wayback YYYYMMDDHHMMSS")
    ap.add_argument("--reason", help="why this window -- shown on the results page")
    ap.add_argument("--clear", action="store_true", help="remove the window entirely")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    await init_db()

    async with async_session_maker() as session:
        project = (await session.execute(
            select(Project).where(Project.id == args.project)
        )).scalar_one_or_none()
        if project is None:
            print(f"no project with id {args.project}", file=sys.stderr)
            return 1

        if not (args.clear or args.start or args.end or args.reason):
            await show(session, project)
            return 0

        if args.clear:
            project.scope_start = project.scope_end = project.scope_reason = None
        else:
            if args.start:
                project.scope_start = args.start
            if args.end:
                project.scope_end = args.end
            if args.reason:
                project.scope_reason = textwrap.fill(
                    " ".join(args.reason.split()), 10_000
                )

        print(f"about to set scope on project {project.id} ({project.domain}):")
        print(f"  start  : {project.scope_start or '(unbounded)'}")
        print(f"  end    : {project.scope_end or '(unbounded)'}")
        print(f"  reason : {project.scope_reason or '(none)'}")
        if not args.yes:
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1

        await session.commit()
        counts = await apply_scope(session, project)
        print(f"\ndone: {counts['in_scope']} captures in scope, "
              f"{counts['excluded']} excluded")
        await session.refresh(project)
        print()
        await show(session, project)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
