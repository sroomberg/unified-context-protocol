"""Unit tests for the UCP SessionEngine core (requires maturin develop)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("ucp._engine")

from ucp._engine import CACHE_THRESHOLD_TOKENS, SessionEngine


def test_cache_threshold_constant() -> None:
    assert CACHE_THRESHOLD_TOKENS == 1024


def test_append_turn_and_token_counts() -> None:
    engine = SessionEngine("You are a test harness.")
    oai, ant, ow = engine.append_turn("user", "Hello from UCP.")
    assert oai > 0
    assert ant > 0
    assert ow > 0
    assert "Human:" in engine.get_master_context()


def test_openai_payload_is_messages_json() -> None:
    engine = SessionEngine("system prompt")
    engine.append_turn("user", "ping")
    engine.append_turn("assistant", "pong")
    payload = json.loads(engine.get_openai_payload())
    assert "messages" in payload
    assert payload["stream"] is True
    roles = [m["role"] for m in payload["messages"]]
    assert "user" in roles
    assert "assistant" in roles
    blob = json.dumps(payload)
    assert "cache_control" not in blob
    assert "ephemeral" not in blob


def test_anthropic_payload_has_ephemeral_cache_control() -> None:
    engine = SessionEngine("system prompt")
    engine.append_turn("user", "alpha")
    engine.append_turn("assistant", "beta")
    engine.append_turn("user", "gamma")
    payload = json.loads(engine.get_anthropic_payload())
    assert payload["stream"] is True
    assert "messages" in payload
    last = payload["messages"][-1]
    content = last["content"]
    assert isinstance(content, list)
    assert content[-1]["cache_control"]["type"] == "ephemeral"


def test_hot_swap_pivots_family_without_clearing_context() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "keep me")
    before = engine.get_master_context()
    fam, tmpl = engine.swap_model("gpt")
    assert fam == "OpenAI"
    assert "ChatML" in tmpl or tmpl == "ChatML"
    assert engine.get_master_context() == before
    fam2, _ = engine.swap_model("openweight", "llama3")
    assert fam2 == "OpenWeight"


def test_openweight_payload_includes_raw_prompt() -> None:
    engine = SessionEngine("sys")
    engine.swap_model("openweight", "chatml")
    engine.append_turn("user", "hi")
    payload = json.loads(engine.get_openweight_payload(True))
    assert "messages" in payload
    assert "raw_prompt" in payload
    assert "<|im_start|>" in payload["raw_prompt"]


def test_cache_warmth_factor_scales() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "x" * 200)
    assert engine.cache_warmth_factor("openai") == 0.0
    oai, _, _ = engine.token_counts()
    engine.set_cache_warm_tokens("openai", oai)
    warmth = engine.cache_warmth_factor("openai")
    assert 0.0 < warmth <= 1.0


def test_char_to_token_index() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "abcdef")
    idx = engine.char_to_token_index("openai", 2)
    assert isinstance(idx, int)
    assert idx >= 0


def test_alignment_table_export() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "token table")
    table = engine.get_alignment_table("openai")
    assert len(table) > 0
    assert "char_start" in table[0]
    assert "token_id" in table[0]


def test_append_text_and_align_releases_and_counts() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "base")
    oai, ant, ow = engine.append_text_and_align(" more text")
    assert oai >= 1 and ant >= 1 and ow >= 1


def test_clear_resets_state() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "bye")
    engine.clear()
    assert engine.get_master_context() == ""
    assert engine.token_counts() == (0, 0, 0)


def test_grok_payload_matches_openai_shape() -> None:
    engine = SessionEngine("sys")
    engine.append_turn("user", "hello grok")
    grok = json.loads(engine.get_grok_payload())
    oai = json.loads(engine.get_openai_payload())
    assert set(grok.keys()) == set(oai.keys())
    assert "messages" in grok
