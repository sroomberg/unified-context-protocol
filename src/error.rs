//! Unified Context Protocol error types.
//!
//! All fallible core APIs return [`UcpResult`]. Optional PyO3 bindings convert
//! these into Python exceptions via `From<UcpError> for PyErr`.

use std::fmt;

/// Errors produced by the UCP core engine.
#[derive(Debug)]
pub enum UcpError {
    /// Tokenizer or engine initialization failed.
    Init(String),
    /// Tokenization or HuggingFace tokenizer load/encode failed.
    Tokenize(String),
    /// Unknown or unsupported model family alias.
    InvalidFamily(String),
    /// Unknown or unsupported prompt template alias.
    InvalidTemplate(String),
    /// Requested tokenizer file path does not exist.
    TokenizerNotFound(String),
    /// JSON payload serialization failed.
    Serialize(String),
}

impl fmt::Display for UcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Init(m)
            | Self::Tokenize(m)
            | Self::InvalidFamily(m)
            | Self::InvalidTemplate(m)
            | Self::TokenizerNotFound(m)
            | Self::Serialize(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for UcpError {}

/// Convenient result alias for UCP core operations.
pub type UcpResult<T> = Result<T, UcpError>;
