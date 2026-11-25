"""LLM processing service using Ollama for HTML content analysis.

Transport lives in llm_client.py; the batch worker lives in processing.py.
What remains here is the analysis itself: prompt construction, HTML cleaning,
response validation, and the domain-card synthesis.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import LLMConfig, get_llm_config, resolve_local_path
from .llm_client import OllamaClient, check_endpoint


@dataclass
class AnalysisResult:
    """Outcome of one analysis attempt.

    A non-None ``error`` means no FileAnalysis row should be written and the
    file should be marked ERROR so it can be retried. The previous code
    fabricated a stub record on a parse failure and marked the file PROCESSED,
    which quietly filled the database with empty analyses.
    """
    analysis: Optional[Dict[str, Any]] = None
    raw_response: str = ""
    tokens: int = 0
    duration: float = 0.0
    error: Optional[str] = None
    fatal: bool = False
    done_reason: Optional[str] = None


def unparseable_error(result: Dict[str, Any], num_predict: Optional[int] = None) -> str:
    """Say why a reply could not be used, in words that point at the fix.

    Ollama reports done_reason="length" when the reply hit num_predict. A
    thinking model (gpt-oss) spends that budget on hidden reasoning first, so
    its JSON is cut off; that was the cause of every Revista Espiral failure,
    and the old generic message gave no hint of it.
    """
    reason = result.get("done_reason")
    if reason == "length":
        limit = num_predict or result.get("eval_count") or "?"
        thinking = " after spending it on hidden reasoning" if result.get("thinking") else ""
        return (f"Cut off at the output limit ({limit} tokens){thinking}. "
                "Thinking models such as gpt-oss need a larger limit; try gemma4.")
    if not (result.get("response") or "").strip():
        return f"Model returned an empty reply (stop reason: {reason or 'unknown'})"
    return f"Model returned an unparseable analysis (stop reason: {reason or 'unknown'})"


def failure_raw(result: Dict[str, Any]) -> str:
    """The text worth keeping from a failed reply: the answer, or failing that
    the model's reasoning, so a cut-off thinking model is visible as such."""
    raw = result.get("response") or ""
    if not raw.strip() and result.get("thinking"):
        return "[no answer; the model's reasoning was:]\n" + result["thinking"]
    return raw


SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
DATE_CONFIDENCES = {"high", "medium", "low", "unknown"}
INTERACTION_TYPES = {
    "discussion", "article", "announcement", "blog_post", "forum_thread",
    "profile", "news", "other",
}
CONTENT_TYPES = {
    "news", "economy", "discussion", "technology", "entertainment",
    "personal", "business", "education", "other",
}

# Small models frequently echo the whole slash-separated option list back
# instead of choosing one ("news/economy/discussion/technology/..."). In the
# existing data that happened on 30 of 52 rows, which makes the results-page
# filters useless. Seeing this many valid options in one value means the model
# copied the menu rather than answering.
_ECHO_THRESHOLD = 3


def _pick_enum(value: Any, allowed: set, default: Optional[str] = None) -> Optional[str]:
    """Coerce a model-supplied enum field to one of the allowed values."""
    if value is None:
        return default
    text = str(value).strip().lower().replace(" ", "_")
    if not text:
        return default
    if text in allowed:
        return text

    parts = [p.strip() for p in re.split(r"[/,|]", text) if p.strip()]
    matches = [p for p in parts if p in allowed]
    if len(matches) >= _ECHO_THRESHOLD:
        return default          # echoed option list, not an answer
    if matches:
        return matches[0]
    return default


def _as_str_list(value: Any, limit: int = 40) -> List[str]:
    """Coerce a model-supplied field into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name") or item.get("person") or item.get("title") or ""
        text = str(item).strip()
        if text:
            out.append(text[:200])
        if len(out) >= limit:
            break
    return out


class LLMService:
    """Service for processing HTML files through Ollama"""

    MAX_FILE_SIZE = 500000  # Skip files larger than 500KB

    # Mimetypes worth sending to a text model. Everything else in the corpus
    # (images, css, js, revisit records) is skipped without an LLM call.
    ANALYZABLE_MIMETYPES = {
        "text/html", "application/xhtml+xml", "text/plain",
        "text/x-component", "text/xml", "application/xml",
        "application/rss+xml", "unk", "", None,
    }

    # ----- health -----
    async def check_ollama(self, cfg: Optional[LLMConfig] = None) -> Dict[str, Any]:
        """Check the configured endpoint and list its models."""
        return await check_endpoint(cfg or get_llm_config())

    # ----- content preparation -----
    def clean_html(self, html_content: str) -> str:
        """Strip HTML tags and clean up content for analysis"""
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        html_content = re.sub(r'\s+', ' ', html_content)
        return html_content.strip()

    def read_and_clean(self, local_path: str, max_chars: int) -> str:
        """Read a file and reduce it to plain text. Synchronous on purpose.

        The regex passes are CPU-bound over documents averaging ~135KB. Run
        directly on the event loop at concurrency 8+ they stall the 2s HTMX
        poll, so callers hand this to asyncio.to_thread().
        """
        path = resolve_local_path(local_path)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        return self.clean_html(content)[:max_chars]

    def build_analysis_prompt(self, content: str, filename: str, original_url: str,
                              max_chars: int = 8000) -> str:
        """Build the prompt for LLM analysis.

        The enum-style fields are pinned to English so the results page filters
        stay consistent, while free-text fields follow the source language --
        this corpus is largely Spanish.
        """
        return f"""Analyze this archived web page content and extract structured information.

File: {filename}
Original URL: {original_url}

Content:
{content[:max_chars]}

Provide your analysis in the following JSON format. Be precise and only include information you can identify from the content:

{{
    "estimated_date": "YYYY-MM-DD or descriptive date if found in content, null if not found",
    "date_confidence": "high/medium/low/unknown",
    "people_mentioned": ["list of names of people mentioned or interacting"],
    "interaction_type": "discussion/article/announcement/blog_post/forum_thread/profile/news/other",
    "subjects": ["main topics or subjects discussed"],
    "sentiment": "positive/negative/neutral/mixed",
    "sentiment_score": 0.0,
    "content_type": "news/economy/discussion/technology/entertainment/personal/business/education/other",
    "tags": ["relevant tags to categorize this content"],
    "description": "2-4 sentence summary of what this page contains"
}}

Notes:
- For sentiment_score: use -1.0 (very negative) to 1.0 (very positive), 0.0 for neutral
- Only include people who are clearly identified by name
- Be specific with tags - they should help categorize and find this content later
- If information cannot be determined, use null or empty arrays
- The values for "date_confidence", "interaction_type", "sentiment" and "content_type"
  must be one of the English options listed above, exactly as written
- Write "description", "subjects" and "tags" in the same language as the page content

Respond ONLY with the JSON object, no additional text."""

    # ----- validation -----
    def validate_analysis(self, data: Any) -> Optional[Dict[str, Any]]:
        """Normalize a model response, or return None if it is unusable.

        Returning None is what routes a bad response to ERROR (retryable)
        instead of persisting an empty record.
        """
        if not isinstance(data, dict):
            return None

        description = data.get("description")
        description = str(description).strip() if description is not None else ""
        if not description:
            return None

        score = data.get("sentiment_score")
        try:
            score = max(-1.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            score = None

        raw_sentiment = data.get("sentiment")
        sentiment = _pick_enum(raw_sentiment, SENTIMENTS)
        # Models sometimes put the number in the label field. Recover it rather
        # than discarding the signal.
        if sentiment is None and score is None and raw_sentiment is not None:
            try:
                score = max(-1.0, min(1.0, float(str(raw_sentiment).strip())))
            except (TypeError, ValueError):
                pass
        if sentiment is None and score is not None:
            sentiment = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"

        estimated_date = data.get("estimated_date")
        estimated_date = str(estimated_date).strip()[:50] if estimated_date else None

        return {
            "estimated_date": estimated_date or None,
            "date_confidence": _pick_enum(data.get("date_confidence"), DATE_CONFIDENCES, "unknown"),
            "people_mentioned": _as_str_list(data.get("people_mentioned")),
            "interaction_type": _pick_enum(data.get("interaction_type"), INTERACTION_TYPES, "other"),
            "subjects": _as_str_list(data.get("subjects")),
            "sentiment": sentiment,
            "sentiment_score": score,
            "content_type": _pick_enum(data.get("content_type"), CONTENT_TYPES, "other"),
            "tags": _as_str_list(data.get("tags")),
            "description": description[:5000],
        }

    @staticmethod
    def _extract_json(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*\}', raw or "")
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    # ----- analysis -----
    async def analyze_content(
        self,
        content: str,
        filename: str,
        original_url: str,
        cfg: LLMConfig,
        client: OllamaClient,
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> AnalysisResult:
        """Send content to Ollama and return a validated analysis."""
        from .llm_client import OllamaError  # local import keeps the module import light

        prompt = self.build_analysis_prompt(content, filename, original_url, cfg.max_content_chars)
        try:
            result = await client.generate(
                prompt,
                model=model or cfg.model,
                options={
                    "num_predict": cfg.num_predict,
                    "temperature": temperature,
                    "top_p": 0.9,
                },
                fmt="json",
            )
        except OllamaError as exc:
            return AnalysisResult(error=cfg.redact(str(exc)), fatal=exc.fatal)
        except Exception as exc:  # transport-level surprise
            return AnalysisResult(error=cfg.redact(f"{type(exc).__name__}: {exc}"))

        raw_response = result.get("response", "") or ""
        tokens = result.get("eval_count", 0) or 0
        duration = (result.get("eval_duration", 0) or 0) / 1_000_000_000

        parsed = self._extract_json(raw_response)
        analysis = self.validate_analysis(parsed)
        done_reason = result.get("done_reason")
        if analysis is None:
            return AnalysisResult(
                raw_response=failure_raw(result),
                tokens=tokens,
                duration=duration,
                error=unparseable_error(result, cfg.num_predict),
                done_reason=done_reason,
            )

        return AnalysisResult(
            analysis=analysis,
            raw_response=raw_response,
            tokens=tokens,
            duration=duration,
            done_reason=done_reason,
        )

    # ----- delegation to the batch worker -----
    # routers/projects.py and routers/web.py both call these; they now forward
    # to processing.py. Imported lazily to avoid a circular import.
    async def start_processing(self, project_id: int, **kwargs) -> Dict[str, Any]:
        from .processing import processing_worker
        return await processing_worker.start(project_id, **kwargs)

    async def stop_processing(self, project_id: int) -> Dict[str, Any]:
        from .processing import processing_worker
        return await processing_worker.stop(project_id)

    async def get_progress(self, project_id: int) -> Dict[str, Any]:
        from .processing import processing_worker
        return await processing_worker.get_progress(project_id)

    # ----- domain card -----
    async def synthesize_domain_card(
        self,
        domain: str,
        model: str,
        analyses: list,
        timeline: Dict[str, Any],
        tech_stack: list,
        whois: Dict[str, Any],
        people: list,
        cfg: Optional[LLMConfig] = None,
    ) -> Dict[str, Any]:
        """Write a short narrative summary of a domain from its aggregated analyses.

        Returns {"summary": str, "model": str}. Falls back to a deterministic
        summary if Ollama is unavailable so the card is still useful offline.
        """
        cfg = cfg or get_llm_config()

        subjects: Dict[str, int] = {}
        content_types: Dict[str, int] = {}
        descriptions = []
        for a in analyses:
            for s in (getattr(a, "subjects", None) or []):
                subjects[s] = subjects.get(s, 0) + 1
            ct = getattr(a, "content_type", None)
            if ct:
                content_types[ct] = content_types.get(ct, 0) + 1
            desc = getattr(a, "description", None)
            if desc:
                descriptions.append(desc)

        top_subjects = sorted(subjects, key=subjects.get, reverse=True)[:15]
        top_content = sorted(content_types, key=content_types.get, reverse=True)[:6]
        top_people = [p["name"] for p in people[:12]]
        tech_names = [t["name"] for t in tech_stack[:15]]
        sample_descriptions = descriptions[:12]

        first = timeline.get("first_seen")
        last = timeline.get("last_seen")

        # Build a deterministic fallback first (also used if the LLM fails).
        fallback_bits = [f"Archived snapshots of {domain}"]
        if first or last:
            fallback_bits.append(f"span {first or '?'} to {last or '?'}")
        if top_content:
            fallback_bits.append(f"Content is mostly {', '.join(top_content)}")
        if top_subjects:
            fallback_bits.append(f"recurring topics include {', '.join(top_subjects[:6])}")
        if tech_names:
            fallback_bits.append(f"Built with {', '.join(tech_names[:6])}")
        if top_people:
            fallback_bits.append(f"Frequently mentioned: {', '.join(top_people[:6])}")
        fallback = ". ".join(fallback_bits) + "."

        prompt = f"""You are writing a concise archival "card" summary for a defunct website,
based on data recovered from the Internet Archive.

Domain: {domain}
Active period (Wayback captures): {first or 'unknown'} to {last or 'unknown'}
Detected technologies: {', '.join(tech_names) or 'unknown'}
Registered domain: {whois.get('whois_domain') or domain} ({'currently registered' if whois.get('whois_source') == 'rdap' else 'not currently registered' if whois.get('whois_source') == 'unregistered' else 'registration unknown'})
Registrar (current WHOIS): {whois.get('registrar') or 'unknown/redacted'}
Dominant content types: {', '.join(top_content) or 'unknown'}
Recurring topics: {', '.join(top_subjects) or 'unknown'}
People frequently named in the content: {', '.join(top_people) or 'none identified'}

Sample page summaries:
- """ + "\n- ".join(sample_descriptions) + """

Write 3-5 sentences describing what this website was: its apparent purpose and community,
the era it was active, the kind of content it hosted, and (only if the evidence supports it)
who appears to have run or been central to it. Be factual and hedge appropriately. Do not
invent specifics. Respond with prose only, no headings or lists."""

        # Cap the card timeout: this runs inside a web request, and the
        # deterministic fallback above is a perfectly good answer.
        from dataclasses import replace
        card_cfg = replace(cfg, request_timeout=min(cfg.request_timeout, 60.0), max_retries=1)

        try:
            async with OllamaClient(card_cfg) as client:
                result = await client.generate(
                    prompt,
                    model=model or card_cfg.model,
                    options={"num_predict": 400, "temperature": 0.5, "top_p": 0.9},
                    fmt=None,
                )
            summary = (result.get("response") or "").strip()
            if not summary:
                return {"summary": fallback, "model": f"{model} (empty response, used fallback)"}
            return {"summary": summary, "model": model}
        except Exception:
            return {"summary": fallback, "model": "fallback (Ollama unavailable)"}


# Global instance
llm_service = LLMService()
