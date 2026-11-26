"""Command-line runner for long jobs that should not depend on a browser tab.

Inside the container:
    python -m app.cli links-extract --project 4
    python -m app.cli links-recover --project 4 [--limit 200]
    python -m app.cli links-analyze --project 4 [--limit 50]
    python -m app.cli links-requeue --project 4 --which unconfirmed|errors|analysis_errors
    python -m app.cli links-stats   --project 4
    python -m app.cli read-select   --project 4          # queue the article-shaped pages
    python -m app.cli read-run      --project 4 [--limit 50]
    python -m app.cli read-requeue  --project 4 --which errors|all
    python -m app.cli read-stats    --project 4
    python -m app.cli comments      --project 4   # reader comments, no AI
    python -m app.cli img-select    --project 5   # images on pages with no text
    python -m app.cli img-recover   --project 5 [--limit 50]
    python -m app.cli img-describe  --project 5 [--limit 50]   # vision model
    python -m app.cli img-requeue   --project 5 --which errors|describe_errors
    python -m app.cli img-stats     --project 5
    python -m app.cli links-memento --project 4 [--limit 50]   # other web archives, no AI
    python -m app.cli img-memento   --project 5 [--limit 50]
    python -m app.cli estimate      --project 4   # calls and tokens each AI job would use
    python -m app.cli search-index  --project 4   # full-text index of the held pages
    python -m app.cli warc          --project 4 [--all]        # WARC of the held captures

Run these against the same database as the web app only while that app is not
running the same job, the in-memory job registry is per process.
"""
import argparse
import logging
import asyncio
import json
import signal

from sqlalchemy import text

from .database import async_session_maker, init_db
from .services.images import image_service
from .services.links import JobState, link_service
from .services.reading import reading_service
from .services.reception import reception_service


async def _stats(project_id: int) -> dict:
    async with async_session_maker() as session:
        rows = (await session.execute(text("""
            SELECT is_internal, kind, recovery_status, analysis_status, COUNT(*)
            FROM link_targets WHERE project_id = :pid
            GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
        """), {"pid": project_id})).fetchall()
        citations = (await session.execute(text(
            "SELECT COUNT(*) FROM page_links WHERE project_id = :pid"), {"pid": project_id})).scalar()
    return {
        "citations": citations,
        "groups": [
            {"scope": "internal" if r[0] else "external", "kind": r[1],
             "recovery": r[2], "analysis": r[3], "count": r[4]}
            for r in rows
        ],
    }


async def _estimate(project_id: int) -> dict:
    """What each LLM job would cost if run now over everything queued."""
    from .database import async_session_maker
    from .models import Project
    from .services import estimate
    out = {}
    async with async_session_maker() as db:
        project = await db.get(Project, project_id)
        model = project.ollama_model if project else None
        for job in estimate.JOBS:
            e = await estimate.estimate(db, project_id, job, model)
            out[job] = {**e, "summary": estimate.describe(e) or "nothing queued"}
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    parser.add_argument("command", choices=["links-extract", "links-recover", "links-analyze",
                                            "links-requeue", "links-stats",
                                            "read-select", "read-run", "read-requeue", "read-stats",
                                            "comments", "img-select", "img-recover",
                                            "img-describe", "img-requeue", "img-stats",
                                            "estimate", "search-index", "warc",
                                            "links-memento", "img-memento"])
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--which", default="unconfirmed")
    parser.add_argument("--all", action="store_true",
                        help="warc: include out-of-scope captures too")
    args = parser.parse_args()

    from .services import applog
    applog.setup("cli")
    logging.getLogger("app.cli").info("cli %s --project %s started", args.command, args.project,
                                       extra={"project_id": args.project})
    await init_db()
    state = JobState(kind=args.command)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, state.stop.set)  # finish the current item, then stop

    async def report():
        while True:
            await asyncio.sleep(10)
            print(f"[{args.command}] {state.done}/{state.total} errors={state.errors} "
                  f"calls={state.calls} {state.message or ''} {state.current or ''}", flush=True)

    reporter = asyncio.create_task(report())
    try:
        if args.command == "links-extract":
            result = await link_service.extract(args.project, state)
        elif args.command == "links-recover":
            result = await link_service.recover(args.project, state, args.limit)
        elif args.command == "links-analyze":
            result = await link_service.analyze(args.project, state, args.limit)
        elif args.command == "links-requeue":
            result = {"requeued": await link_service.requeue(args.project, args.which)}
        elif args.command == "links-stats":
            result = await _stats(args.project)
        elif args.command == "read-select":
            result = await reading_service.select_pages(args.project)
        elif args.command == "read-run":
            result = await reading_service.read(args.project, state, args.limit)
        elif args.command == "comments":
            result = await reception_service.extract(args.project, state)
        elif args.command == "img-select":
            result = await image_service.select(args.project)
        elif args.command == "img-recover":
            result = await image_service.recover(args.project, state, args.limit)
        elif args.command == "img-describe":
            result = await image_service.describe(args.project, state, args.limit)
        elif args.command == "img-requeue":
            result = {"requeued": await image_service.requeue(args.project, args.which)}
        elif args.command == "img-stats":
            result = await image_service.stats(args.project)
        elif args.command == "links-memento":
            from .services.memento import recover_links
            result = await recover_links(args.project, state, args.limit)
        elif args.command == "img-memento":
            from .services.memento import recover_images
            result = await recover_images(args.project, state, args.limit)
        elif args.command == "warc":
            from .database import async_session_maker
            from .models import Project
            from .services.warc_export import build_warc
            async with async_session_maker() as db:
                result = await build_warc(db, await db.get(Project, args.project), all_captures=args.all)
        elif args.command == "search-index":
            from .services.search_index import index_project
            result = await index_project(args.project)
        elif args.command == "estimate":
            result = await _estimate(args.project)
        elif args.command == "read-requeue":
            result = {"requeued": await reading_service.requeue(args.project, args.which)}
        else:
            result = await reading_service.stats(args.project)
    finally:
        reporter.cancel()
    if state.message:
        print(state.message)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
