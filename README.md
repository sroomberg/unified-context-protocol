# Unified Context Protocol (UCP) *EXPERIMENTAL*

**Rust-first infrastructure** for linear cross-model context, tokenizer alignment,
and provider payload transposition.

Python bindings are **optional**. UCP does not include a REPL, TUI, HTTP client,
or agent loop — those belong in separate runtimes (OpenCode, custom agents, etc.).

## What UCP owns

| Concern | Mechanism |
|---|---|
| Append-only master context | `SessionEngine` (Rust) |
| Cross-tokenizer offset maps | tiktoken + HF/approx tables |
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
│  UCP core (Rust crate `ucp`)             │
│  SessionEngine · payloads · alignment    │
└───────────────────┬──────────────────────┘
                    │ optional feature `python`
┌───────────────────▼──────────────────────┐
│  Optional bindings (`python/ucp`, PyO3)  │
└──────────────────────────────────────────┘
```

## Layout

```
Cargo.toml                 Rust crate (default: no Python)
src/
  lib.rs                   crate root
  engine.rs                pure SessionEngine
  error.rs
  python.rs                optional PyO3 module (feature = "python")
python/ucp/                optional Python package helpers
pyproject.toml             maturin packaging (optional)
tests/                     Python binding tests (optional)
```

## Rust usage (primary)

```bash
cargo test
cargo build --release
# with optional Python extension:
cargo build --features python
```

```rust
use ucp::SessionEngine;

let engine = SessionEngine::new(Some("You are helpful.".into()))?;
engine.append_turn("user".into(), "Explain prompt caching.".into())?;
let body = engine.get_anthropic_payload()?;
engine.swap_model("openai", None)?;
let openai_body = engine.get_openai_payload()?;
```

## Optional Python bindings

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --extras dev   # enables Cargo feature `python`
pytest -q
```

```python
from ucp import SessionEngine, ProviderId, build_request_body, ProtocolSettings

settings = ProtocolSettings()
engine = SessionEngine(settings.system_prompt)
engine.append_turn("user", "Explain prompt caching.")
body = build_request_body(ProviderId.ANTHROPIC, engine, settings, model="claude-3-5-sonnet-20241022")
engine.swap_model("openai")
```

Optional tokenizer paths: `UCP_ANTHROPIC_TOKENIZER_PATH` /
`UCP_OPENWEIGHT_TOKENIZER_PATH` (see `.env.example`).

## Changelog & releases

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/). See [`CHANGELOG.md`](CHANGELOG.md).

**Pull requests** must either:

1. Add an entry under `## [Unreleased]` in `CHANGELOG.md`, or
2. Use a [Conventional Commits](https://www.conventionalcommits.org/) PR title
   (`feat:`, `fix:`, `docs:`, …), or
3. Carry the `skip-changelog` label

**Releases** are automated by [release-please](https://github.com/googleapis/release-please):

1. Merges to `master` open/update a **Release PR** with version bumps + CHANGELOG
2. Merging that Release PR creates the GitHub Release and `vX.Y.Z` tag

## License

MIT — see `LICENSE`.
