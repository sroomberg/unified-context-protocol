"""Provider family IDs and pure payload transposition helpers.

No HTTP clients. Runtimes / agents call these to obtain provider-native
JSON bodies from a ``SessionEngine`` master context.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, MutableMapping, Optional

from ucp.config import ProtocolSettings


class ProviderId(str, Enum):
    """Logical model family addressed by UCP payloads."""

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
    def engine_family(self) -> str:
        """Name accepted by ``SessionEngine.swap_model``."""
        return {
            ProviderId.OPENAI: "openai",
            ProviderId.ANTHROPIC: "anthropic",
            ProviderId.GROK: "grok",
            ProviderId.OPENWEIGHT: "openweight",
        }[self]


def build_request_body(
    provider: ProviderId | str,
    engine: Any,
    settings: ProtocolSettings | None = None,
    *,
    model: str | None = None,
    stream: bool = True,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Transpose ``SessionEngine`` master context into a provider-native JSON body.

    Anthropic payloads include ``cache_control: {type: ephemeral}`` on the last
    history block (emitted by the Rust core, re-checked here). OpenAI / Grok /
    open-weight bodies are Chat Completions shaped; open-weight may include a
    ``raw_prompt`` plus an optional prefix-cache hint for vLLM-style servers.

    Credentials, base URLs, and HTTP transport are intentionally omitted — those
    belong to the consuming runtime.
    """
    pid = provider if isinstance(provider, ProviderId) else ProviderId.parse(provider)
    cfg = settings or ProtocolSettings()

    if pid is ProviderId.ANTHROPIC:
        body: MutableMapping[str, Any] = json.loads(engine.get_anthropic_payload())
        if model:
            body["model"] = model
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body["stream"] = stream
        if body.get("system") is None:
            body.pop("system", None)
        ensure_anthropic_cache_control(body)
        return dict(body)

    if pid is ProviderId.OPENWEIGHT:
        body = json.loads(engine.get_openweight_payload(True))
        if model:
            body["model"] = model
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body["stream"] = stream
        if cfg.openweight_enable_prefix_cache_hint:
            body["extra_body"] = {"prefix_caching": True}
        return dict(body)

    raw = (
        engine.get_grok_payload()
        if pid is ProviderId.GROK
        else engine.get_openai_payload()
    )
    body = json.loads(raw)
    if model:
        body["model"] = model
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    body["stream"] = stream
    return dict(body)


def ensure_anthropic_cache_control(body: MutableMapping[str, Any]) -> None:
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
