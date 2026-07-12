# Unified Context Protocol (UCP)

**Infrastructure only** — linear cross-model context, tokenizer alignment, and
provider payload transposition. UCP does **not** include a REPL, TUI, HTTP
client, or agent loop. Those belong in separate runtimes (OpenCode, custom
agents, gateways, etc.).

## What UCP owns

| Concern | Mechanism |
|---|---|
| Append-only master context | `SessionEngine` (Rust / PyO3) |
| Cross-tokenizer offset maps | tiktoken + HF/approx tables, GIL-released |
| Hot-swap at the semantic boundary | `swap_model` without rewriting history |
| Provider-native payloads | ChatML / Anthropic blocks / Llama3 / Mistral |
| Anthropic prompt-cache breakpoints | `cache_control: { type: ephemeral }` on last block |
| Cache warmth accounting | `set_cache_warm_tokens` / `cache_warmth_factor` |

## What UCP does **not** own

- API keys, base URLs, retries, streaming SSE
- Dual-write network fan-out (runtime responsibility)
- Tool calling, file edits, git, or agent UX
- A terminal UI

## Architecture

```
┌──────────────────────────────────────────┐
│  Runtime / agent (OpenCode, custom, …)   │
│  HTTP · tools · UX · dual-write policy   │
└───────────────────┬──────────────────────┘
                    │ uses
┌───────────────────▼──────────────────────┐
│  UCP infra (this repo)                   │
│  SessionEngine · ProviderId · payloads   │
└──────────────────────────────────────────┘
```

## Layout

```
Cargo.toml / pyproject.toml
src/lib.rs                 SessionEngine PyO3 module
python/ucp/
  __init__.py
  config.py                ProtocolSettings (no secrets)
  providers.py             pure payload helpers
tests/
tokenizers/                optional HF tokenizer.json drop-in
.env.example
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --extras dev
```

Optional: set `UCP_ANTHROPIC_TOKENIZER_PATH` / `UCP_OPENWEIGHT_TOKENIZER_PATH`
to a HuggingFace `tokenizer.json` for precise alignment. Without them, OpenAI
uses tiktoken; Anthropic/open-weight use a deterministic approx tokenizer.

## Library usage

```python
from ucp import SessionEngine, ProviderId, build_request_body, ProtocolSettings

settings = ProtocolSettings()
engine = SessionEngine(settings.system_prompt)
engine.append_turn("system", settings.system_prompt)
engine.append_turn("user", "Explain prompt caching.")

# Runtime performs HTTP; UCP only shapes the body.
body = build_request_body(ProviderId.ANTHROPIC, engine, settings, model="claude-3-5-sonnet-20241022")

# Mid-session semantic swap — history stays linear.
engine.swap_model("openai")
openai_body = build_request_body("gpt", engine, model="gpt-5.5")

# Runtime reports cache hits back into warmth accounting.
oai, ant, ow = engine.token_counts()
engine.set_cache_warm_tokens("anthropic", ant)
print(engine.cache_warmth_factor("anthropic"))
```

Open-weight / local OpenAI-compatible servers are first-class via
`ProviderId.OPENWEIGHT` (ChatML, Llama3, or Mistral templates).

## Development

```bash
cargo test
maturin develop --extras dev
pytest -q
```

## License

MIT — see `LICENSE`.
