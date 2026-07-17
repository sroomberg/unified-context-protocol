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

/// Crate version from `Cargo.toml` (SemVer).
///
/// Kept in sync with `pyproject.toml` / `ucp.__version__` by release-please.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod version_tests {
    use super::VERSION;

    #[test]
    fn version_is_semver() {
        let parts: Vec<_> = VERSION.split('.').collect();
        assert!(parts.len() >= 3, "expected MAJOR.MINOR.PATCH, got {VERSION}");
        for part in &parts[..3] {
            assert!(part.parse::<u64>().is_ok(), "non-numeric SemVer component in {VERSION}");
        }
    }
}

#[cfg(feature = "python")]
mod python;
