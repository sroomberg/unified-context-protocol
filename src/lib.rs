//! Unified Context Protocol (UCP) — Rust-first context alignment infra.
//!
//! # Overview
//!
//! UCP keeps a **linear, append-only** master conversation context and maintains
//! cross-tokenizer offset maps so a consuming runtime can hot-swap between
//! OpenAI, Anthropic, Grok, and open-weight providers without rewriting history.
//!
//! This crate intentionally has **no network I/O and no agent UX**. Optional
//! PyO3 bindings are available behind the `python` Cargo feature (used by
//! `maturin` / the `python/ucp` package).
//!
//! # Example
//!
//! ```rust,no_run
//! use ucp::SessionEngine;
//!
//! let engine = SessionEngine::new(Some("You are helpful.".into()))?;
//! engine.append_turn("user".into(), "Explain prompt caching.".into())?;
//! let anthropic_body = engine.get_anthropic_payload()?;
//! engine.swap_model("openai", None)?;
//! let openai_body = engine.get_openai_payload()?;
//! # Ok::<(), ucp::UcpError>(())
//! ```

mod engine;
mod error;

pub use engine::{
    default_template_for, parse_family, parse_template, ModelFamily, PromptTemplate, SessionEngine,
    TokenSpan, CACHE_THRESHOLD_TOKENS,
};
pub use error::{UcpError, UcpResult};

#[cfg(feature = "python")]
mod python;
