# Cross-Model Hot-Swapping REPL Harness

Lightweight zero-token-burn LLM REPL that hot-swaps mid-conversation between
**OpenAI GPT-5.5**, **Anthropic Claude 3.5 Sonnet**, **Grok (xAI)**, and
**open-weight / local** OpenAI-compatible servers (vLLM, Ollama, LM Studio, …).

Cross-provider GPU KV caches cannot be shared. This harness instead keeps a
**linear append-only master context**, dual-writes prefixes asynchronously so
each provider’s prompt cache stays warm, and uses a Rust **cross-tokenizer
offset mapping engine** so swaps happen at the semantic boundary with no
prefill recalculation tax on cache hits.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Textual TUI (app.py)                                       │
│  chat feed · status sidebar · /swap macros · stream cursor  │
└────────────────────────────┬────────────────────────────────┘
                             │ asyncio
┌────────────────────────────▼────────────────────────────────┐
│  NetworkPipeline (network.py)                               │
│  dual_write_warmup via asyncio.gather                       │
│  Anthropic cache_control ephemeral · OpenAI/Grok auto-cache │
│  Open-weight prefix_caching hints (vLLM / SGLang)           │
└────────────────────────────┬────────────────────────────────┘
                             │ PyO3
┌────────────────────────────▼────────────────────────────────┐
│  SessionEngine (src/lib.rs)                                 │
│  master_context · tiktoken + HF/approx tables · rayon       │
│  GIL release (py.allow_threads) · ChatML / Anthropic / Llama│
└─────────────────────────────────────────────────────────────┘
```

### Hot-swap strategy

| Provider | Cache trigger | Discount (typical) | Harness behavior |
|---|---|---|---|
| OpenAI GPT-5.5 | Matching prefix ≥ 1,024 tokens | ~50% + fast TTFT | Automatic; keep prefix byte-stable |
| Anthropic Claude 3.5 | Explicit `cache_control: ephemeral` on a block ≥ 1,024 tokens | ~90% on hits | Last history block tagged; 5-minute TTL |
| Grok (xAI) | OpenAI-compatible prefix caching | provider-dependent | Same payload path as OpenAI |
| Open-weight (vLLM, …) | Server automatic prefix cache | free / local | `extra_body.prefix_caching` + ChatML/Llama/Mistral templates |

## Project layout

```
Cargo.toml / pyproject.toml   Rust + maturin / Python packaging
src/lib.rs                    SessionEngine PyO3 module
python/hotswap_repl/
  app.py                      Textual TUI
  network.py                  Dual-write + streaming
  providers.py                OpenAI / Anthropic / Grok / open-weight
  config.py                   pydantic-settings
  metrics.py                  Cache warmth factor
  cli.py                      `hotswap-repl` entrypoint
tests/                        Engine + network tests
tokenizers/                   Optional HF tokenizer.json drop-in
.env.example                  API keys and endpoints
```

## Quick start

```bash
# System deps: Rust toolchain, Python ≥ 3.10, maturin
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --extras dev

cp .env.example .env   # add OPENAI_API_KEY / ANTHROPIC_API_KEY / XAI_API_KEY

hotswap-repl           # or: python -m hotswap_repl.cli ui
```

### Open-weight local example (Ollama)

```bash
ollama serve
ollama pull llama3.1
# .env
OPENWEIGHT_BASE_URL=http://127.0.0.1:11434/v1
OPENWEIGHT_MODEL=llama3.1
OPENWEIGHT_TEMPLATE=llama3
DEFAULT_PROVIDER=openweight
```

Optional: drop a HuggingFace `tokenizer.json` into `tokenizers/` and set
`OPENWEIGHT_TOKENIZER_PATH` / `ANTHROPIC_TOKENIZER_PATH` for precise alignment.
Without those files the Rust core uses a deterministic approx tokenizer for
Anthropic / open-weight spans while OpenAI uses `tiktoken` (o200k / cl100k).

## TUI commands

| Command | Action |
|---|---|
| `/swap gpt` | Hot-swap to OpenAI |
| `/swap claude` | Hot-swap to Anthropic |
| `/swap grok` | Hot-swap to xAI Grok |
| `/swap open` | Hot-swap to open-weight / local |
| `/warmup` | Speculative dual-write now |
| `/stats` | Engine alignment stats |
| `/clear` | Reset master context |
| `/help` | Help |
| `Ctrl+S` | Cycle providers |
| `Ctrl+C` | Quit |

On `/swap`, the UI blinks, `SessionEngine.swap_model` pivots templates instantly,
and the streaming cursor retargets to the alternate API route without rewriting
history.

## CLI utilities

```bash
hotswap-repl dump-payload anthropic --user "hello"
hotswap-repl dump-payload openweight --user "hello"
hotswap-repl warmup --user "long system document…"
```

## Development

```bash
cargo test
maturin develop --extras dev
pytest -q
```

## License

MIT — see `LICENSE`.
