//! Unified Context Protocol (UCP) — Rust-first context alignment infra.
//!
//! The core API is pure Rust. Optional PyO3 bindings are available behind the
//! `python` Cargo feature (used by `maturin` / the `python/ucp` package).

mod engine;
mod error;

pub use engine::{
    default_template_for, parse_family, parse_template, ModelFamily, PromptTemplate, SessionEngine,
    TokenSpan, CACHE_THRESHOLD_TOKENS,
};
pub use error::{UcpError, UcpResult};

#[cfg(feature = "python")]
mod python;
