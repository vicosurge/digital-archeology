"""Application log: one JSON object per line under <data>/logs/, read back by the Logs page.

The web server and the CLI are separate processes, so each writes its own file
(web.log, cli.log) and the page merges them. Every record passes through the
LLM config's redaction first, so the API key never reaches disk even when an
exception message happens to contain a request header.

Per-file failures also stay on their database rows (error_message); the log
is the timeline: what ran, when, and what went wrong around it.
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from ..config import get_data_dir, get_llm_config

LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
_configured = False


def log_dir() -> str:
    path = os.path.join(get_data_dir(), "logs")
    os.makedirs(path, exist_ok=True)
    return path


class _JsonFormatter(logging.Formatter):
    def __init__(self, redact):
        super().__init__()
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        out: Dict[str, Any] = {
            "t": datetime.fromtimestamp(record.created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": self._redact(msg),
        }
        project = getattr(record, "project_id", None)
        if project is not None:
            out["project"] = project
        return json.dumps(out, ensure_ascii=False)


class _Quiet(logging.Filter):
    """Drop chatter that would bury the useful lines (HTTP client request lines)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name.startswith(("httpx", "httpcore")) and record.levelno < logging.WARNING)


def setup(process: str) -> None:
    """Attach the JSON file handler. Safe to call more than once."""
    global _configured
    if _configured:
        return
    _configured = True
    try:
        redact = get_llm_config().redact
    except Exception:          # config errors must not stop logging
        redact = lambda s: s   # noqa: E731
    handler = RotatingFileHandler(os.path.join(log_dir(), f"{process}.log"),
                                  maxBytes=2_000_000, backupCount=4, encoding="utf-8")
    handler.setFormatter(_JsonFormatter(redact))
    handler.addFilter(_Quiet())
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    # uvicorn's error logger (tracebacks of failed requests) does not
    # propagate to the root logger, so it gets the handler directly.
    for name in ("uvicorn.error",):
        lg = logging.getLogger(name)
        if handler not in lg.handlers:
            lg.addHandler(handler)


def _tail(path: str, max_lines: int) -> List[str]:
    """Last lines of a file without reading all of it."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block, data = 64 * 1024, b""
        while size > 0 and data.count(b"\n") <= max_lines:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    return data.decode("utf-8", "replace").splitlines()[-max_lines:]


def read(limit: int = 300, level: str = "info", q: str = "",
         project: Optional[int] = None, scan: int = 5000) -> List[Dict[str, Any]]:
    """Newest-first records from every process's current and last rotated file."""
    floor = LEVELS.get(level, 20)
    needle = q.lower().strip()
    records: List[Dict[str, Any]] = []
    d = log_dir()
    for name in sorted(os.listdir(d)):
        # Only this module's files; scripts write their own plain-text logs here too.
        if name not in ("web.log", "web.log.1", "cli.log", "cli.log.1"):
            continue
        for line in _tail(os.path.join(d, name), scan):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if LEVELS.get(r.get("level"), 20) < floor:
                continue
            if project is not None and r.get("project") != project:
                continue
            if needle and needle not in (r.get("msg", "") + r.get("logger", "")).lower():
                continue
            r["process"] = name.split(".")[0]
            records.append(r)
    records.sort(key=lambda r: r.get("t", ""), reverse=True)
    return records[:limit]


def counts_since(hours: int = 24) -> Dict[str, int]:
    """Warnings and errors in the last ``hours``, for the nav badge."""
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    out = {"warning": 0, "error": 0}
    for r in read(limit=10_000, level="warning", scan=3000):
        try:
            ts = datetime.strptime(r["t"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
        except (KeyError, ValueError):
            continue
        if ts < cutoff:
            break
        if r["level"] in out:
            out[r["level"]] += 1
    return out
