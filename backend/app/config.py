"""Application settings.

Everything the LLM pipeline needs to reach Ollama lives here, read from the
environment. Settings are resolved *per use* rather than at import time, so a
container restart is enough to repoint the app at a different endpoint -- the
previous code captured the host once at import and required a code reload.

The API key is treated as a secret: never print a config object directly, and
run any message that might embed the endpoint or key through ``redact()``
before it reaches a log line, a template, or the database.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    """Resolved connection settings for one Ollama endpoint."""

    base_url: str
    api_key: str
    auth_header: str
    auth_scheme: str
    model: str
    connect_timeout: float
    request_timeout: float
    max_retries: int
    backoff_base: float
    concurrency: int
    requests_per_minute: int
    max_content_chars: int
    num_predict: int
    # Budget controls. A metered endpoint (ollama.com) enforces a session quota
    # that resets on a rolling window; these let a run stop cleanly at a chosen
    # point and be resumed in the next window.
    max_calls_per_run: int          # 0 = unlimited
    stop_at_session_usage: float    # 0 = disabled, else a fraction like 0.80

    @property
    def has_auth(self) -> bool:
        return bool(self.api_key)

    def headers(self) -> dict:
        """Auth headers for the endpoint, or {} when no key is configured."""
        if not self.api_key:
            return {}
        value = f"{self.auth_scheme} {self.api_key}".strip() if self.auth_scheme else self.api_key
        return {self.auth_header: value}

    def safe_base_url(self) -> str:
        """The endpoint with any embedded credentials stripped, safe to display."""
        try:
            parts = urlsplit(self.base_url)
        except ValueError:
            return "(unparseable URL)"
        if not parts.hostname:
            return self.base_url
        netloc = parts.hostname
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))

    def redact(self, text: Optional[str]) -> str:
        """Strip the API key and any URL userinfo out of a message.

        Applied to every error string that leaves the client, because those
        strings end up in ``archived_files.error_message`` and in HTML title
        attributes where a leaked token would persist.
        """
        if not text:
            return ""
        out = str(text)
        if self.api_key:
            out = out.replace(self.api_key, "***")
        password = os.environ.get("DA_PASSWORD", "").strip()
        if password:
            out = out.replace(password, "***")
        if self.base_url and self.base_url != self.safe_base_url():
            out = out.replace(self.base_url, self.safe_base_url())
        return out

    def __repr__(self) -> str:  # pragma: no cover - defensive against accidental logging
        return (
            f"LLMConfig(base_url={self.safe_base_url()!r}, model={self.model!r}, "
            f"auth={'yes' if self.has_auth else 'no'}, concurrency={self.concurrency})"
        )


def load_llm_config() -> LLMConfig:
    """Build an LLMConfig from the environment.

    ``OLLAMA_HOST`` is kept as the primary name so existing .env files and
    docker-compose setups keep working untouched.
    """
    return LLMConfig(
        base_url=_env_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        api_key=_env_str("OLLAMA_API_KEY", ""),
        auth_header=_env_str("OLLAMA_AUTH_HEADER", "Authorization"),
        # Set OLLAMA_AUTH_SCHEME="" to send the bare key (X-Api-Key style).
        auth_scheme=os.environ.get("OLLAMA_AUTH_SCHEME", "Bearer").strip(),
        model=_env_str("OLLAMA_MODEL", "llama3.2:3b"),
        connect_timeout=_env_float("LLM_CONNECT_TIMEOUT", 10.0),
        request_timeout=_env_float("LLM_REQUEST_TIMEOUT", 180.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 3),
        backoff_base=_env_float("LLM_BACKOFF_BASE", 1.5),
        concurrency=max(1, _env_int("LLM_CONCURRENCY", 4)),
        requests_per_minute=max(0, _env_int("LLM_REQUESTS_PER_MINUTE", 0)),
        max_content_chars=_env_int("LLM_MAX_CONTENT_CHARS", 8000),
        num_predict=_env_int("LLM_NUM_PREDICT", 500),
        max_calls_per_run=max(0, _env_int("LLM_MAX_CALLS_PER_RUN", 0)),
        stop_at_session_usage=max(0.0, min(1.0, _env_float("LLM_STOP_AT_SESSION_USAGE", 0.0))),
    )


def get_prices() -> tuple:
    """(input, output) dollars per million tokens; (0, 0) when not configured."""
    return (max(0.0, _env_float("LLM_PRICE_INPUT_PER_MTOK", 0.0)),
            max(0.0, _env_float("LLM_PRICE_OUTPUT_PER_MTOK", 0.0)))


def get_auth() -> Optional[tuple]:
    """(username, password) when the UI is password-protected, else None.

    Read on each request, like the LLM settings. Set DA_PASSWORD whenever anyone
    but you can reach the port: without it the app has no login at all.
    """
    password = os.environ.get("DA_PASSWORD", "").strip()
    if not password:
        return None
    return _env_str("DA_USERNAME", "archive"), password


def get_llm_config() -> LLMConfig:
    """Current LLM settings. Resolved on each call, so no restart-to-reconfigure."""
    return load_llm_config()


# ----- Data location -----

@lru_cache(maxsize=1)
def _default_data_dir() -> str:
    # backend/app/config.py -> backend/data
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_data_dir() -> str:
    """Directory holding the SQLite database and the downloaded corpus.

    Overridable with DA_DATA_DIR. This matters because ``local_path`` values in
    the database are absolute *container* paths (/app/data/...): outside Docker
    they resolve to nothing, so both the DB and the corpus have to be located
    explicitly. See resolve_local_path().
    """
    return _env_str("DA_DATA_DIR", _default_data_dir())


def resolve_local_path(path: Optional[str]) -> Optional[str]:
    """Map a stored local_path onto the current data directory.

    Files were recorded from inside the container as /app/data/projects/...
    When the app runs elsewhere, rebase that prefix onto get_data_dir() so the
    same corpus is readable without rewriting every row.
    """
    if not path:
        return path
    data_dir = os.path.abspath(get_data_dir())
    if os.path.isabs(path):
        for prefix in ("/app/data", "/data"):
            if path.startswith(prefix + os.sep):
                return os.path.join(data_dir, path[len(prefix) + 1:])
        if path.startswith(data_dir + os.sep):
            return path
        # An absolute path from some other layout: try the tail after /projects/.
        marker = os.sep + "projects" + os.sep
        idx = path.find(marker)
        if idx != -1:
            return os.path.join(data_dir, path[idx + 1:])
        return path
    return os.path.join(data_dir, path)
