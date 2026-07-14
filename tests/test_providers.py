"""Tests for UCP protocol helpers (pure payload transposition)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("ucp._engine")

from ucp._engine import SessionEngine
from ucp.config import ProtocolSettings
from ucp.providers import (
    ProviderId,
    build_request_body,
    ensure_anthropic_cache_control,
)


def test_provider_id_aliases() -> None:
    assert ProviderId.parse("gpt") is ProviderId.OPENAI
    assert ProviderId.parse("claude") is ProviderId.ANTHROPIC
    assert ProviderId.parse("xai") is ProviderId.GROK
    assert ProviderId.parse("ollama") is ProviderId.OPENWEIGHT
    with pytest.raises(ValueError):
        ProviderId.parse("nope")


def test_build_anthropic_body_decorates_cache_control() -> None:
    settings = ProtocolSettings()
    engine = SessionEngine(settings.system_prompt)
    engine.append_turn("user", "hello")
    body = build_request_body(
        ProviderId.ANTHROPIC,
        engine,
        settings,
        model="claude-3-5-sonnet-20241022",
        stream=False,
    )
    assert body["model"] == "claude-3-5-sonnet-20241022"
    last = body["messages"][-1]["content"][-1]
    assert last["cache_control"]["type"] == "ephemeral"


def test_ensure_cache_control_on_string_content() -> None:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "plain"}],
    }
    ensure_anthropic_cache_control(body)
    block = body["messages"][0]["content"][0]
    assert block["cache_control"]["type"] == "ephemeral"
    assert block["text"] == "plain"


def test_build_openweight_body_includes_prefix_hint() -> None:
    settings = ProtocolSettings(openweight_enable_prefix_cache_hint=True)
    engine = SessionEngine("sys")
    engine.append_turn("user", "local model")
    body = build_request_body(
        ProviderId.OPENWEIGHT,
        engine,
        settings,
        model="llama3.1",
        stream=True,
    )
    assert body["stream"] is True
    assert body.get("extra_body", {}).get("prefix_caching") is True


def test_build_openai_body_strips_anthropic_flags() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "hi")
    body = build_request_body(ProviderId.OPENAI, engine, model="gpt-5.5")
    blob = str(body)
    assert "cache_control" not in blob
    assert body["model"] == "gpt-5.5"
    assert "messages" in body


def test_swap_then_build_uses_engine_family() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "cross")
    engine.swap_model(ProviderId.GROK.engine_family)
    body = build_request_body("grok", engine, model="grok-3")
    assert body["model"] == "grok-3"
    assert "messages" in body
