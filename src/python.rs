//! Optional PyO3 bindings for UCP (`ucp._engine`).
//!
//! Enabled with `--features python`. Thin wrappers over the pure Rust
//! [`crate::SessionEngine`] that release the GIL during tokenization.

// PyO3 + `From<UcpError> for PyErr` trips `useless_conversion` on recent clippy
// when pymethods return `PyResult` after `?` / `Into` conversion.
#![allow(clippy::useless_conversion)]

use crate::engine::{SessionEngine, CACHE_THRESHOLD_TOKENS};
use crate::error::{UcpError, UcpResult};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

impl From<UcpError> for PyErr {
    fn from(err: UcpError) -> Self {
        match err {
            UcpError::InvalidFamily(m)
            | UcpError::InvalidTemplate(m)
            | UcpError::TokenizerNotFound(m) => PyValueError::new_err(m),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}

/// Python-facing handle around the Rust [`SessionEngine`].
#[pyclass(name = "SessionEngine")]
pub struct PySessionEngine {
    inner: Arc<SessionEngine>,
}

#[pymethods]
impl PySessionEngine {
    #[new]
    #[pyo3(signature = (system_prompt=None))]
    fn new(system_prompt: Option<String>) -> UcpResult<Self> {
        Ok(Self {
            inner: Arc::new(SessionEngine::new(system_prompt)?),
        })
    }

    /// Append text and realign tokenizers (GIL released).
    fn append_text_and_align(
        &self,
        py: Python<'_>,
        text: String,
    ) -> UcpResult<(usize, usize, usize)> {
        let engine = Arc::clone(&self.inner);
        py.allow_threads(move || engine.append_text_and_align(text))
    }

    #[pyo3(signature = (role, content))]
    fn append_turn(
        &self,
        py: Python<'_>,
        role: String,
        content: String,
    ) -> UcpResult<(usize, usize, usize)> {
        let engine = Arc::clone(&self.inner);
        py.allow_threads(move || engine.append_turn(role, content))
    }

    fn get_openai_payload(&self) -> UcpResult<String> {
        self.inner.get_openai_payload()
    }

    fn get_anthropic_payload(&self) -> UcpResult<String> {
        self.inner.get_anthropic_payload()
    }

    fn get_grok_payload(&self) -> UcpResult<String> {
        self.inner.get_grok_payload()
    }

    #[pyo3(signature = (include_raw_prompt=true))]
    fn get_openweight_payload(&self, include_raw_prompt: bool) -> UcpResult<String> {
        self.inner.get_openweight_payload(include_raw_prompt)
    }

    #[pyo3(signature = (family, template=None))]
    fn swap_model(&self, family: &str, template: Option<&str>) -> UcpResult<(String, String)> {
        self.inner.swap_model(family, template)
    }

    fn get_active_model(&self) -> String {
        self.inner.get_active_model()
    }

    fn get_master_context(&self) -> String {
        self.inner.get_master_context()
    }

    fn get_system_prompt(&self) -> String {
        self.inner.get_system_prompt()
    }

    fn set_system_prompt(&self, prompt: String) {
        self.inner.set_system_prompt(prompt);
    }

    fn token_counts(&self) -> (usize, usize, usize) {
        self.inner.token_counts()
    }

    fn cache_warmth_factor(&self, provider: &str) -> f64 {
        self.inner.cache_warmth_factor(provider)
    }

    fn set_cache_warm_tokens(&self, provider: &str, tokens: usize) {
        self.inner.set_cache_warm_tokens(provider, tokens);
    }

    #[pyo3(signature = (path, target="anthropic"))]
    fn load_hf_tokenizer(&self, path: String, target: &str) -> UcpResult<String> {
        self.inner.load_hf_tokenizer(path, target)
    }

    fn get_alignment_table<'py>(
        &self,
        py: Python<'py>,
        family: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let spans = self.inner.get_alignment_table(family)?;
        let list = PyList::empty_bound(py);
        for span in spans {
            let d = PyDict::new_bound(py);
            d.set_item("char_start", span.char_start)?;
            d.set_item("char_end", span.char_end)?;
            d.set_item("token_id", span.token_id)?;
            list.append(d)?;
        }
        Ok(list)
    }

    fn char_to_token_index(&self, family: &str, char_index: usize) -> usize {
        self.inner.char_to_token_index(family, char_index)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.stats();
        let d = PyDict::new_bound(py);
        for (k, v) in stats {
            d.set_item(k, json_to_py(py, &v)?)?;
        }
        Ok(d)
    }

    fn clear(&self) {
        self.inner.clear();
    }
}

#[pyclass(eq, eq_int, name = "ModelFamily")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyModelFamily {
    OpenAI = 0,
    Anthropic = 1,
    Grok = 2,
    OpenWeight = 3,
}

#[pymethods]
impl PyModelFamily {
    fn __repr__(&self) -> String {
        format!("ModelFamily.{:?}", self)
    }
}

impl std::fmt::Debug for PyModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI => write!(f, "OpenAI"),
            Self::Anthropic => write!(f, "Anthropic"),
            Self::Grok => write!(f, "Grok"),
            Self::OpenWeight => write!(f, "OpenWeight"),
        }
    }
}

#[pyclass(eq, eq_int, name = "PromptTemplate")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyPromptTemplate {
    ChatML = 0,
    AnthropicBlocks = 1,
    Llama3 = 2,
    MistralInstruct = 3,
    Plain = 4,
}

#[pymethods]
impl PyPromptTemplate {
    fn __repr__(&self) -> String {
        format!("PromptTemplate.{:?}", self)
    }
}

impl std::fmt::Debug for PyPromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChatML => write!(f, "ChatML"),
            Self::AnthropicBlocks => write!(f, "AnthropicBlocks"),
            Self::Llama3 => write!(f, "Llama3"),
            Self::MistralInstruct => write!(f, "MistralInstruct"),
            Self::Plain => write!(f, "Plain"),
        }
    }
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_py(py))
            } else {
                Ok(n.as_f64().unwrap_or(0.0).into_py(py))
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        other => Ok(other.to_string().into_py(py)),
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySessionEngine>()?;
    m.add_class::<PyModelFamily>()?;
    m.add_class::<PyPromptTemplate>()?;
    m.add("CACHE_THRESHOLD_TOKENS", CACHE_THRESHOLD_TOKENS)?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
