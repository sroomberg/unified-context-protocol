//! Unified Context Protocol error types.

use std::fmt;

#[derive(Debug)]
pub enum UcpError {
    Init(String),
    Tokenize(String),
    InvalidFamily(String),
    InvalidTemplate(String),
    TokenizerNotFound(String),
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

pub type UcpResult<T> = Result<T, UcpError>;
