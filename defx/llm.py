"""The only module that talks to the network.

OpenAI SDK pointed at the Portkey gateway (option A in the brief), structured output
validated against defx.schema.Extraction, explicit retries with backoff, and a disk
cache so that re-runs and resumed runs cost nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

from .schema import Extraction

DEFAULT_BASE_URL = "https://api.portkey.ai/v1"
DEFAULT_MODEL = "gpt-4.1"
CACHE_DIR = Path(".cache")
# Largest response seen: 11,420 tokens for a 31-defendant caption (evidence quotes dominate).
# Models differ in their ceiling (gpt-4.1: 32,768; gpt-4o and gpt-4o-mini: 16,384), so this is a
# request, lowered automatically to what the model reports it supports.
MAX_OUTPUT_TOKENS = 32_000
_TOKEN_LIMIT_RE = re.compile(r"supports at most (\d+) completion tokens")

# USD per 1M tokens (input, output), list prices; used only for the cost estimate.
PRICES = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class ExtractionError(RuntimeError):
    """The model could not produce a valid extraction for a document."""


class ConfigurationError(RuntimeError):
    """Wrong key, model or base URL. Every document would fail the same way, so the run stops."""


FATAL = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


@dataclass
class CallResult:
    extraction: Extraction
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    retries: int = 0  # transport-level: rate limit, timeout, 5xx
    schema_repairs: int = 0  # output-level: refusal or invalid JSON, asked again


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    name = model.split("/")[-1].lstrip("@")
    for known in sorted(PRICES, key=len, reverse=True):
        if name.startswith(known):
            price_in, price_out = PRICES[known]
            return (prompt_tokens * price_in + completion_tokens * price_out) / 1e6
    return 0.0


def backoff_seconds(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter."""
    return random.uniform(0, min(cap, base * 2**attempt))


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        use_cache: bool = True,
        max_attempts: int = 5,
        timeout_s: float = 180.0,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        api_key = os.environ.get("PORTKEY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("PORTKEY_API_KEY is not set (export it or put it in .env)")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.use_cache = use_cache
        self.max_attempts = max_attempts
        self.cache_dir = cache_dir
        self.structured = True  # falls back to JSON mode if the gateway rejects a JSON schema
        self.max_output_tokens = MAX_OUTPUT_TOKENS
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout_s,
            max_retries=0,  # retries are handled here so they can be counted
        )

    # -- cache ----------------------------------------------------------------------------

    def _cache_path(self, messages: list[dict]) -> Path:
        payload = json.dumps(
            [self.model, messages, Extraction.model_json_schema()], sort_keys=True, ensure_ascii=False
        )
        return self.cache_dir / (hashlib.sha256(payload.encode("utf-8")).hexdigest() + ".json")

    def _read_cache(self, path: Path) -> CallResult | None:
        if not (self.use_cache and path.exists()):
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CallResult(
                extraction=Extraction.model_validate(data["extraction"]),
                prompt_tokens=data["prompt_tokens"],
                completion_tokens=data["completion_tokens"],
                latency_s=data["latency_s"],
                cached=True,
            )
        except (ValueError, KeyError):  # corrupt entry: ignore and call the model again
            return None

    def _write_cache(self, path: Path, result: CallResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "extraction": result.extraction.model_dump(),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_s": result.latency_s,
                    "model": self.model,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written entry

    # -- call -----------------------------------------------------------------------------

    async def _request(self, messages: list[dict]) -> tuple[Extraction | None, object]:
        """One API call. Returns (parsed or None, usage)."""
        common = dict(model=self.model, messages=messages, temperature=0, seed=0, max_tokens=self.max_output_tokens)
        if self.structured:
            completion = await self._client.chat.completions.parse(response_format=Extraction, **common)
            message = completion.choices[0].message
            return (None if message.refusal else message.parsed), completion.usage
        schema_hint = {
            "role": "system",
            "content": "Reply with one JSON object matching this JSON schema:\n"
            + json.dumps(Extraction.model_json_schema()),
        }
        completion = await self._client.chat.completions.create(
            response_format={"type": "json_object"}, **{**common, "messages": [schema_hint, *messages]}
        )
        try:
            return Extraction.model_validate_json(completion.choices[0].message.content or ""), completion.usage
        except ValidationError:
            return None, completion.usage

    async def extract(self, messages: list[dict]) -> CallResult:
        path = self._cache_path(messages)
        cached = self._read_cache(path)
        if cached:
            return cached

        retries = schema_repairs = prompt_tokens = completion_tokens = 0
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                parsed, usage = await self._request(messages)
            except RETRYABLE as exc:
                last_error = exc
                retries += 1
                await asyncio.sleep(backoff_seconds(attempt))
                continue
            except FATAL as exc:
                raise ConfigurationError(f"model '{self.model}' is not usable with this key: {exc}") from exc
            except openai.BadRequestError as exc:
                limit = _TOKEN_LIMIT_RE.search(str(exc))
                if limit:  # model with a smaller output ceiling; concurrent calls all land here once
                    self.max_output_tokens = min(self.max_output_tokens, int(limit.group(1)))
                    continue
                if self.structured and "response_format" in str(exc):
                    self.structured = False  # gateway/model without JSON-schema support
                    continue
                raise ExtractionError(f"bad request: {exc}") from exc
            except openai.LengthFinishReasonError as exc:
                # truncated at max_tokens; output length varies between calls, so one more try is worthwhile
                last_error = ExtractionError("output truncated at max_tokens")
                schema_repairs += 1
                if schema_repairs > 1:
                    raise last_error from exc
                continue

            if usage is not None:
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
            if parsed is None:  # refusal or JSON that fails validation: ask again
                last_error = ExtractionError("model returned no valid extraction")
                schema_repairs += 1
                continue

            result = CallResult(
                extraction=parsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=time.monotonic() - started,
                retries=retries,
                schema_repairs=schema_repairs,
            )
            self._write_cache(path, result)
            return result
        raise ExtractionError(f"gave up after {self.max_attempts} attempts: {last_error!r}")

    async def close(self) -> None:
        await self._client.close()
