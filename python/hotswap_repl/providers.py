"""Provider registry and request builders for proprietary + open-weight models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Mapping, MutableMapping, Optional

import httpx

from hotswap_repl.config import Settings


class ProviderId(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"
    OPENWEIGHT = "openweight"

    @classmethod
    def parse(cls, value: str) -> "ProviderId":
        key = value.strip().lower().lstrip("/")
        aliases: dict[str, ProviderId] = {
            "openai": cls.OPENAI,
            "gpt": cls.OPENAI,
            "gpt-5.5": cls.OPENAI,
            "gpt5.5": cls.OPENAI,
            "oai": cls.OPENAI,
            "anthropic": cls.ANTHROPIC,
            "claude": cls.ANTHROPIC,
            "sonnet": cls.ANTHROPIC,
            "claude-3.5": cls.ANTHROPIC,
            "grok": cls.GROK,
            "xai": cls.GROK,
            "openweight": cls.OPENWEIGHT,
            "open": cls.OPENWEIGHT,
            "local": cls.OPENWEIGHT,
            "ollama": cls.OPENWEIGHT,
            "vllm": cls.OPENWEIGHT,
            "hf": cls.OPENWEIGHT,
        }
        if key not in aliases:
            raise ValueError(
                f"Unknown provider '{value}'. "
                f"Expected one of: {', '.join(sorted(set(aliases)))}"
            )
        return aliases[key]

    @property
    def display_name(self) -> str:
        return {
            ProviderId.OPENAI: "GPT-5.5 (OpenAI)",
            ProviderId.ANTHROPIC: "Claude 3.5 Sonnet (Anthropic)",
            ProviderId.GROK: "Grok (xAI)",
            ProviderId.OPENWEIGHT: "Open-Weight / Local",
        }[self]

    @property
    def engine_family(self) -> str:
        """Name accepted by SessionEngine.swap_model."""
        return {
            ProviderId.OPENAI: "openai",
            ProviderId.ANTHROPIC: "anthropic",
            ProviderId.GROK: "grok",
            ProviderId.OPENWEIGHT: "openweight",
        }[self]


@dataclass(frozen=True)
class ProviderEndpoint:
    provider: ProviderId
    base_url: str
    model: str
    api_key: str
    supports_anthropic_cache: bool = False
    supports_openai_cache: bool = False
    supports_prefix_cache: bool = False
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        if self.provider is ProviderId.ANTHROPIC:
            return f"{self.base_url.rstrip('/')}/v1/messages"
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        headers.update(dict(self.extra_headers))
        if self.provider is ProviderId.ANTHROPIC:
            if self.api_key:
                headers["x-api-key"] = self.api_key
            headers.setdefault("anthropic-version", "2023-06-01")
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ProviderRegistry:
    """Resolves configured endpoints for every supported provider family."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._endpoints: dict[ProviderId, ProviderEndpoint] = {
            ProviderId.OPENAI: ProviderEndpoint(
                provider=ProviderId.OPENAI,
                base_url=settings.openai_base_url,
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                supports_openai_cache=True,
            ),
            ProviderId.ANTHROPIC: ProviderEndpoint(
                provider=ProviderId.ANTHROPIC,
                base_url=settings.anthropic_base_url,
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                supports_anthropic_cache=True,
                extra_headers={"anthropic-version": settings.anthropic_version},
            ),
            ProviderId.GROK: ProviderEndpoint(
                provider=ProviderId.GROK,
                base_url=settings.grok_base_url,
                model=settings.grok_model,
                api_key=settings.grok_api_key,
                supports_openai_cache=True,
            ),
            ProviderId.OPENWEIGHT: ProviderEndpoint(
                provider=ProviderId.OPENWEIGHT,
                base_url=settings.openweight_base_url,
                model=settings.openweight_model,
                api_key=settings.openweight_api_key or "EMPTY",
                supports_prefix_cache=settings.openweight_enable_prefix_cache,
            ),
        }

    def get(self, provider: ProviderId | str) -> ProviderEndpoint:
        pid = provider if isinstance(provider, ProviderId) else ProviderId.parse(provider)
        return self._endpoints[pid]

    def all_ids(self) -> list[ProviderId]:
        return list(self._endpoints.keys())

    def configured_for_warmup(self) -> list[ProviderId]:
        """Providers that have enough config to attempt a speculative warmup."""
        ready: list[ProviderId] = []
        for pid, ep in self._endpoints.items():
            if pid is ProviderId.OPENWEIGHT:
                # Local servers often need no key.
                ready.append(pid)
                continue
            if ep.api_key:
                ready.append(pid)
        return ready


def build_request_body(
    provider: ProviderId,
    engine: Any,
    settings: Settings,
    *,
    stream: bool = True,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Transpose SessionEngine master context into a provider-native JSON body.

    Anthropic payloads already include ``cache_control: {type: ephemeral}`` on
    the last history block (emitted by the Rust core). OpenAI / Grok / open-weight
    payloads are Chat Completions style and rely on automatic prefix caching.
    """
    max_out = max_tokens if max_tokens is not None else settings.max_output_tokens

    if provider is ProviderId.ANTHROPIC:
        raw = engine.get_anthropic_payload()
        body: MutableMapping[str, Any] = json.loads(raw)
        body["model"] = settings.anthropic_model
        body["max_tokens"] = max_out
        body["stream"] = stream
        # Ensure system is a plain string (Anthropic accepts str or text blocks).
        if body.get("system") is None:
            body.pop("system", None)
        _ensure_anthropic_cache_control(body)
        return dict(body)

    if provider is ProviderId.OPENWEIGHT:
        raw = engine.get_openweight_payload(True)
        body = json.loads(raw)
        body["model"] = settings.openweight_model
        body["max_tokens"] = max_out
        body["stream"] = stream
        # Prefer messages API; keep raw_prompt for servers that need it.
        if settings.openweight_enable_prefix_cache:
            # vLLM automatic prefix caching is server-side; hint via extra body.
            body["extra_body"] = {"prefix_caching": True}
        return dict(body)

    # OpenAI + Grok share Chat Completions shape.
    raw = (
        engine.get_grok_payload()
        if provider is ProviderId.GROK
        else engine.get_openai_payload()
    )
    body = json.loads(raw)
    body["model"] = (
        settings.grok_model if provider is ProviderId.GROK else settings.openai_model
    )
    body["max_tokens"] = max_out
    body["stream"] = stream
    return dict(body)


def _ensure_anthropic_cache_control(body: MutableMapping[str, Any]) -> None:
    """Guarantee ephemeral cache_control on the final content block (5-minute TTL)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    last = messages[-1]
    if not isinstance(last, dict):
        return
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        return
    if isinstance(content, list) and content:
        block = content[-1]
        if isinstance(block, dict):
            block["cache_control"] = {"type": "ephemeral"}


async def iter_sse_text(
    response: httpx.Response,
    provider: ProviderId,
) -> AsyncIterator[str]:
    """Yield decoded text deltas from an SSE chat/messages stream."""
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        text = _extract_delta(event, provider)
        if text:
            yield text


def _extract_delta(event: Mapping[str, Any], provider: ProviderId) -> str:
    if provider is ProviderId.ANTHROPIC:
        etype = event.get("type")
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                return str(delta.get("text") or "")
        return ""

    # OpenAI-compatible (OpenAI, Grok, open-weight, vLLM, Ollama).
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            return str(content)
        # Some open-weight servers send message.content on non-stream chunks.
        message = choice.get("message") or {}
        if message.get("content"):
            return str(message["content"])
    return ""
