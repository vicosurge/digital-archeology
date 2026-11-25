"""HTTP client for Ollama.

Split out of llm.py so the transport concerns (auth, retries, timeouts, error
classification, secret redaction) sit in one place and can be reasoned about
independently of the analysis logic.

The important behavioural change over the previous inline client is that a
non-2xx response now raises. Previously the body was parsed regardless, which
turned an auth failure or a 500 into an empty string, then into an empty
"analysis" record that was written to the database and marked PROCESSED. That
is silent data corruption, and at tens of thousands of files it is invisible
until someone reads the results.
"""
import asyncio
import random
from typing import Any, Dict, List, Optional

import httpx

from ..config import LLMConfig


class OllamaError(Exception):
    """A call to Ollama failed.

    ``retryable`` marks transient transport-level problems. ``fatal`` marks
    conditions where continuing the whole run is pointless (bad credentials, a
    model that does not exist) -- retrying those across thousands of files just
    burns time and, on a metered endpoint, money.
    """

    def __init__(self, message: str, *, status: Optional[int] = None,
                 retryable: bool = False, fatal: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.fatal = fatal


class OllamaAuthError(OllamaError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message, status=status, retryable=False, fatal=True)


class RateLimiter:
    """Minimum-interval gate. A limit of 0 disables it entirely."""

    def __init__(self, per_minute: int):
        self._interval = 60.0 / per_minute if per_minute and per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_at = max(now, self._next_at) + self._interval


class OllamaClient:
    """Pooled async client for one Ollama endpoint.

    Holds a single httpx.AsyncClient for the lifetime of a run. The previous
    code opened a fresh client per file, which against a remote HTTPS endpoint
    means a new TLS handshake for every one of thousands of requests.
    """

    def __init__(self, cfg: LLMConfig, limiter: Optional[RateLimiter] = None):
        self.cfg = cfg
        self.limiter = limiter or RateLimiter(cfg.requests_per_minute)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OllamaClient":
        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            headers=self.cfg.headers(),
            timeout=httpx.Timeout(self.cfg.request_timeout, connect=self.cfg.connect_timeout),
            limits=httpx.Limits(
                max_connections=self.cfg.concurrency + 2,
                max_keepalive_connections=self.cfg.concurrency + 2,
            ),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient must be used as an async context manager")
        return self._client

    def _classify(self, response: httpx.Response) -> OllamaError:
        status = response.status_code
        try:
            body = response.text[:400]
        except Exception:
            body = ""
        msg = self.cfg.redact(f"Ollama returned HTTP {status}: {body}".strip())

        if status in (401, 403):
            return OllamaAuthError(
                f"Authentication failed (HTTP {status}). Check OLLAMA_API_KEY.", status=status
            )
        if status == 404 and "model" in body.lower():
            # A missing model will fail identically for every file in the run.
            return OllamaError(
                f"Model not found on the endpoint: {msg}", status=status, fatal=True
            )
        if status == 429 or status >= 500 or status in (408, 425):
            return OllamaError(msg, status=status, retryable=True)
        return OllamaError(msg, status=status)

    @staticmethod
    def _retry_after(response: httpx.Response) -> Optional[float]:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    async def _post_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._require_client()
        attempts = self.cfg.max_retries + 1
        last: Optional[OllamaError] = None

        for attempt in range(attempts):
            await self.limiter.acquire()
            delay: Optional[float] = None
            try:
                response = await client.post(path, json=payload)
                if response.status_code >= 400:
                    err = self._classify(response)
                    if err.fatal or not err.retryable:
                        raise err
                    last = err
                    delay = self._retry_after(response)
                else:
                    return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = OllamaError(
                    self.cfg.redact(f"{type(exc).__name__}: {exc}"), retryable=True
                )

            if attempt < attempts - 1:
                if delay is None:
                    delay = self.cfg.backoff_base ** attempt + random.uniform(0, 0.5)
                await asyncio.sleep(delay)

        raise last or OllamaError("Ollama request failed for an unknown reason")

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        fmt: Optional[str] = "json",
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if fmt:
            payload["format"] = fmt
        if images:
            payload["images"] = images   # base64, for vision models
        return await self._post_with_retry("/api/generate", payload)

    async def fetch_usage(self) -> Optional[Dict[str, float]]:
        """Current quota consumption, if the endpoint reports it.

        ollama.com serves /api/usage with usage as fractions of the plan's
        allowances. Which allowances it reports has changed over time (session
        and weekly in mid-2026, a single monthly one by September), so every
        block under ``limits`` is read and ``max`` carries the highest fraction:
        that is the one a run has to respect. Returns None when the endpoint has
        no such endpoint (a self-hosted Ollama), so callers must treat it as
        optional.
        """
        client = self._require_client()
        try:
            response = await client.get("/api/usage", timeout=httpx.Timeout(15.0))
            if response.status_code >= 400:
                return None
            limits = (response.json() or {}).get("limits") or {}
        except Exception:
            return None

        out: Dict[str, float] = {}
        for key, block in limits.items():
            if not isinstance(block, dict):
                continue
            usage = block.get("usage")
            if isinstance(usage, (int, float)):
                out[key] = float(usage)
            requests = block.get("models") or []
            if requests:
                out[f"{key}_requests"] = float(
                    sum(m.get("request_count", 0) for m in requests)
                )
        fractions = [v for k, v in out.items() if not k.endswith("_requests")]
        if fractions:
            out["max"] = max(fractions)
        return out or None

    async def list_models(self) -> List[str]:
        client = self._require_client()
        response = await client.get(
            "/api/tags", timeout=httpx.Timeout(15.0, connect=self.cfg.connect_timeout)
        )
        if response.status_code >= 400:
            raise self._classify(response)
        data = response.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]


async def check_endpoint(cfg: LLMConfig) -> Dict[str, Any]:
    """Probe the configured endpoint. Never raises; never leaks the key."""
    try:
        async with OllamaClient(cfg) as client:
            models = await client.list_models()
        return {
            "status": "running",
            "models": models,
            "host": cfg.safe_base_url(),
            "authenticated": cfg.has_auth,
        }
    except Exception as exc:
        return {
            "status": "error",
            "models": [],
            "host": cfg.safe_base_url(),
            "authenticated": cfg.has_auth,
            "message": cfg.redact(str(exc)) or type(exc).__name__,
        }
