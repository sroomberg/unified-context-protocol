"""Tests for Python network / provider helpers (mocked HTTP)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("hotswap_repl._engine")

from hotswap_repl._engine import SessionEngine
from hotswap_repl.config import Settings
from hotswap_repl.metrics import collect_metrics, format_warmth
from hotswap_repl.network import NetworkPipeline
from hotswap_repl.providers import (
    ProviderId,
    ProviderRegistry,
    _ensure_anthropic_cache_control,
    build_request_body,
)


def test_provider_id_aliases() -> None:
    assert ProviderId.parse("gpt") is ProviderId.OPENAI
    assert ProviderId.parse("claude") is ProviderId.ANTHROPIC
    assert ProviderId.parse("xai") is ProviderId.GROK
    assert ProviderId.parse("ollama") is ProviderId.OPENWEIGHT
    with pytest.raises(ValueError):
        ProviderId.parse("nope")


def test_build_anthropic_body_decorates_cache_control() -> None:
    settings = Settings(
        ANTHROPIC_API_KEY="test",
        OPENAI_API_KEY="test",
        dual_write_enabled=False,
    )
    engine = SessionEngine(settings.system_prompt)
    engine.append_turn("user", "hello")
    body = build_request_body(ProviderId.ANTHROPIC, engine, settings, stream=False)
    assert body["model"] == settings.anthropic_model
    last = body["messages"][-1]["content"][-1]
    assert last["cache_control"]["type"] == "ephemeral"


def test_ensure_cache_control_on_string_content() -> None:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "plain"}],
    }
    _ensure_anthropic_cache_control(body)
    block = body["messages"][0]["content"][0]
    assert block["cache_control"]["type"] == "ephemeral"
    assert block["text"] == "plain"


def test_build_openweight_body_includes_prefix_hint() -> None:
    settings = Settings(OPENWEIGHT_ENABLE_PREFIX_CACHE=True, dual_write_enabled=False)
    engine = SessionEngine("sys")
    engine.append_turn("user", "local model")
    body = build_request_body(ProviderId.OPENWEIGHT, engine, settings, stream=True)
    assert body["stream"] is True
    assert body.get("extra_body", {}).get("prefix_caching") is True


def test_metrics_and_warmth_formatting() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "metrics")
    engine.set_cache_warm_tokens("anthropic", 50)
    m = collect_metrics(engine, ProviderId.OPENAI)
    assert m.anthropic_tokens > 0
    assert 0.0 <= m.warmth_anthropic <= 1.0
    assert "%" in format_warmth(m.warmth_anthropic)


@pytest.mark.asyncio
async def test_dual_write_warmup_gather() -> None:
    settings = Settings(
        OPENAI_API_KEY="sk-test",
        ANTHROPIC_API_KEY="sk-ant-test",
        XAI_API_KEY="xai-test",
        dual_write_enabled=True,
    )
    engine = SessionEngine(settings.system_prompt)
    engine.append_turn("user", "warmup document " * 40)
    pipeline = NetworkPipeline(
        engine=engine,
        settings=settings,
        registry=ProviderRegistry(settings),
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"usage": {"prompt_tokens": 128}}
    mock_response.text = ""

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.object(pipeline, "_require_client", return_value=mock_client):
        results = await pipeline.dual_write_warmup(
            providers=[ProviderId.OPENAI, ProviderId.ANTHROPIC, ProviderId.GROK]
        )

    assert len(results) == 3
    assert all(r.ok for r in results)
    assert engine.cache_warmth_factor("openai") > 0.0
    assert mock_client.post.await_count == 3


@pytest.mark.asyncio
async def test_set_active_swaps_engine() -> None:
    settings = Settings(dual_write_enabled=False)
    engine = SessionEngine("sys")
    pipeline = NetworkPipeline(engine=engine, settings=settings)
    pid = pipeline.set_active("gpt")
    assert pid is ProviderId.OPENAI
    assert engine.get_active_model() == "OpenAI"


@pytest.mark.asyncio
async def test_stream_completion_yields_chunks() -> None:
    settings = Settings(OPENAI_API_KEY="sk-test", dual_write_enabled=False, default_provider="openai")
    engine = SessionEngine("sys")
    engine.append_turn("user", "stream please")
    pipeline = NetworkPipeline(engine=engine, settings=settings)
    pipeline.set_active("openai")

    lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":"!"}}]}',
        "data: [DONE]",
    ]

    class FakeStream:
        status_code = 200

        async def aiter_lines(self):
            for line in lines:
                yield line

        async def aread(self) -> bytes:
            return b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=FakeStream())

    with patch.object(pipeline, "_require_client", return_value=mock_client):
        chunks = []
        async for chunk in pipeline.stream_completion():
            chunks.append(chunk)

    texts = [c.text for c in chunks if c.text]
    assert texts == ["Hi", "!"]
    assert chunks[-1].done is True
