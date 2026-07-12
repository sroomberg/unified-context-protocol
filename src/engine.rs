//! Pure Rust SessionEngine — append-only master context + dual tokenizer alignment.
//!
//! No Python / network / agent UX. Optional PyO3 bindings live behind the `python` feature.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};
use tokenizers::Tokenizer;

use crate::error::{UcpError, UcpResult};

/// Minimum tokens typically required before provider prompt caches activate.
pub const CACHE_THRESHOLD_TOKENS: usize = 1024;

static CHATML_SPLIT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>").expect("chatml regex")
});

static ANTHROPIC_SPLIT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)(?:^|\n)(System|Human|Assistant)\s*:\s*(.*?)(?=(?:\n(?:System|Human|Assistant)\s*:)|\z)")
        .expect("anthropic regex")
});

/// Absolute character span mapped to a single model token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSpan {
    pub char_start: usize,
    pub char_end: usize,
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    OpenAI = 0,
    Anthropic = 1,
    Grok = 2,
    OpenWeight = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptTemplate {
    ChatML = 0,
    AnthropicBlocks = 1,
    Llama3 = 2,
    MistralInstruct = 3,
    Plain = 4,
}

#[derive(Debug, Clone)]
struct TokenizerBackend {
    #[allow(dead_code)]
    name: String,
    kind: BackendKind,
}

#[derive(Clone)]
enum BackendKind {
    Tiktoken(Arc<CoreBPE>),
    HuggingFace(Arc<Tokenizer>),
    /// Deterministic whitespace / punctuation fallback used when no HF file is loaded.
    FallbackApprox,
}

impl std::fmt::Debug for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Tiktoken(_) => write!(f, "Tiktoken"),
            BackendKind::HuggingFace(_) => write!(f, "HuggingFace"),
            BackendKind::FallbackApprox => write!(f, "FallbackApprox"),
        }
    }
}

#[derive(Debug, Default)]
struct AlignmentTable {
    spans: Vec<TokenSpan>,
}

impl AlignmentTable {
    fn token_count(&self) -> usize {
        self.spans.len()
    }

    fn clear(&mut self) {
        self.spans.clear();
    }
}

#[derive(Debug, Clone)]
struct MessageTurn {
    role: String,
    content: String,
    #[allow(dead_code)]
    char_start: usize,
    #[allow(dead_code)]
    char_end: usize,
}

/// Thread-safe dual-tokenizer session engine (pure Rust API).
pub struct SessionEngine {
    inner: Arc<RwLock<SessionState>>,
}

#[derive(Debug)]
struct SessionState {
    master_context: String,
    openai_table: AlignmentTable,
    anthropic_table: AlignmentTable,
    openweight_table: AlignmentTable,
    openai_backend: TokenizerBackend,
    anthropic_backend: TokenizerBackend,
    openweight_backend: TokenizerBackend,
    active_family: ModelFamily,
    active_template: PromptTemplate,
    openweight_template: PromptTemplate,
    system_prompt: String,
    turns: Vec<MessageTurn>,
    /// Extra registered open-weight backends keyed by alias.
    extra_backends: HashMap<String, TokenizerBackend>,
    extra_tables: HashMap<String, AlignmentTable>,
    /// Last known cache-warmed token counts per family (updated by the consuming runtime).
    cache_warm_tokens: HashMap<String, usize>,
}

impl SessionState {
    fn new() -> UcpResult<Self> {
        let openai_bpe = o200k_base()
            .or_else(|_| cl100k_base())
            .map_err(|e| UcpError::Init(format!("tiktoken init failed: {e}")))?;

        Ok(Self {
            master_context: String::new(),
            openai_table: AlignmentTable::default(),
            anthropic_table: AlignmentTable::default(),
            openweight_table: AlignmentTable::default(),
            openai_backend: TokenizerBackend {
                name: "openai-o200k".into(),
                kind: BackendKind::Tiktoken(Arc::new(openai_bpe)),
            },
            anthropic_backend: TokenizerBackend {
                name: "anthropic-approx".into(),
                kind: BackendKind::FallbackApprox,
            },
            openweight_backend: TokenizerBackend {
                name: "openweight-approx".into(),
                kind: BackendKind::FallbackApprox,
            },
            active_family: ModelFamily::Anthropic,
            active_template: PromptTemplate::AnthropicBlocks,
            openweight_template: PromptTemplate::ChatML,
            system_prompt: String::new(),
            turns: Vec::new(),
            extra_backends: HashMap::new(),
            extra_tables: HashMap::new(),
            cache_warm_tokens: HashMap::from([
                ("openai".into(), 0usize),
                ("anthropic".into(), 0usize),
                ("grok".into(), 0usize),
                ("openweight".into(), 0usize),
            ]),
        })
    }
}

/// Encode text into TokenSpans with absolute character offsets.
fn encode_with_backend(
    backend: &TokenizerBackend,
    text: &str,
    base_offset: usize,
) -> Result<Vec<TokenSpan>, String> {
    match &backend.kind {
        BackendKind::Tiktoken(bpe) => {
            let tokens = bpe.encode_with_special_tokens(text);
            // tiktoken does not expose per-token char spans directly; reconstruct via decode walk.
            let mut spans = Vec::with_capacity(tokens.len());
            let mut cursor = 0usize;
            for &tid in &tokens {
                let piece = bpe
                    .decode(vec![tid])
                    .unwrap_or_else(|_| "�".to_string());
                let start = base_offset + cursor;
                // Prefer exact substring match at cursor; fall back to piece length in chars.
                let piece_chars = piece.chars().count();
                let end = start + piece_chars;
                // Advance cursor by matching against remaining text when possible.
                let remaining: String = text.chars().skip(cursor).collect();
                if remaining.starts_with(&piece) {
                    cursor += piece_chars;
                } else {
                    // Best-effort: advance by decoded piece length.
                    cursor += piece_chars;
                }
                spans.push(TokenSpan {
                    char_start: start,
                    char_end: end.min(base_offset + text.chars().count()),
                    token_id: tid as u32,
                });
            }
            // Clamp final end to actual text length.
            if let Some(last) = spans.last_mut() {
                last.char_end = base_offset + text.chars().count();
            }
            Ok(spans)
        }
        BackendKind::HuggingFace(tok) => {
            let encoding = tok
                .encode(text, false)
                .map_err(|e| format!("HF encode error: {e}"))?;
            let ids = encoding.get_ids();
            let offsets = encoding.get_offsets();
            let mut spans = Vec::with_capacity(ids.len());
            for (i, &id) in ids.iter().enumerate() {
                let (s, e) = offsets.get(i).copied().unwrap_or((0, 0));
                // HF offsets are byte offsets; convert to char indices.
                let char_start = byte_to_char(text, s) + base_offset;
                let char_end = byte_to_char(text, e) + base_offset;
                spans.push(TokenSpan {
                    char_start,
                    char_end,
                    token_id: id,
                });
            }
            Ok(spans)
        }
        BackendKind::FallbackApprox => Ok(approx_tokenize(text, base_offset)),
    }
}

fn byte_to_char(text: &str, byte_idx: usize) -> usize {
    if byte_idx >= text.len() {
        return text.chars().count();
    }
    text[..byte_idx].chars().count()
}

/// Approximate tokenizer: split on whitespace and punctuation, assign stable ids.
fn approx_tokenize(text: &str, base_offset: usize) -> Vec<TokenSpan> {
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\s+|[^\w\s]+|\w+").expect("approx tokenizer"));
    let mut spans = Vec::new();
    let mut char_cursor = 0usize;
    let mut byte_cursor = 0usize;
    for mat in RE.find_iter(text) {
        // Sync char cursor with byte start.
        let start_byte = mat.start();
        while byte_cursor < start_byte {
            let ch = text[byte_cursor..].chars().next().unwrap();
            byte_cursor += ch.len_utf8();
            char_cursor += 1;
        }
        let piece = mat.as_str();
        let char_start = base_offset + char_cursor;
        let piece_chars = piece.chars().count();
        let char_end = char_start + piece_chars;
        let token_id = stable_hash_u32(piece);
        spans.push(TokenSpan {
            char_start,
            char_end,
            token_id,
        });
        char_cursor += piece_chars;
        byte_cursor = mat.end();
    }
    spans
}

fn stable_hash_u32(s: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for b in s.as_bytes() {
        h ^= u32::from(*b);
        h = h.wrapping_mul(16777619);
    }
    h
}

fn realign_all(state: &mut SessionState) -> Result<(), String> {
    let text = state.master_context.clone();
    let jobs: Vec<(&str, TokenizerBackend)> = vec![
        ("openai", state.openai_backend.clone()),
        ("anthropic", state.anthropic_backend.clone()),
        ("openweight", state.openweight_backend.clone()),
    ];

    let results: Vec<Result<(String, Vec<TokenSpan>), String>> = jobs
        .into_par_iter()
        .map(|(name, backend)| {
            let spans = encode_with_backend(&backend, &text, 0)?;
            Ok((name.to_string(), spans))
        })
        .collect();

    for result in results {
        let (name, spans) = result?;
        match name.as_str() {
            "openai" => state.openai_table.spans = spans,
            "anthropic" => state.anthropic_table.spans = spans,
            "openweight" => state.openweight_table.spans = spans,
            _ => {}
        }
    }

    // Extra registered open-weight aliases.
    let extras: Vec<(String, TokenizerBackend)> = state
        .extra_backends
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let extra_results: Vec<Result<(String, Vec<TokenSpan>), String>> = extras
        .into_par_iter()
        .map(|(name, backend)| {
            let spans = encode_with_backend(&backend, &text, 0)?;
            Ok((name, spans))
        })
        .collect();
    for result in extra_results {
        let (name, spans) = result?;
        state
            .extra_tables
            .entry(name)
            .or_default()
            .spans = spans;
    }

    Ok(())
}

fn parse_turns_from_master(master: &str, system_prompt: &str) -> Vec<MessageTurn> {
    // Prefer ChatML if present.
    if master.contains("<|im_start|>") {
        let mut turns = Vec::new();
        for cap in CHATML_SPLIT.captures_iter(master) {
            let role = cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
            let content = cap.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();
            let whole = cap.get(0).unwrap();
            let char_start = master[..whole.start()].chars().count();
            let char_end = char_start + whole.as_str().chars().count();
            turns.push(MessageTurn {
                role,
                content,
                char_start,
                char_end,
            });
        }
        return turns;
    }

    if ANTHROPIC_SPLIT.is_match(master) {
        let mut turns = Vec::new();
        for cap in ANTHROPIC_SPLIT.captures_iter(master) {
            let role = cap
                .get(1)
                .map(|m| m.as_str().to_lowercase())
                .unwrap_or_else(|| "user".into());
            let content = cap
                .get(2)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            let whole = cap.get(0).unwrap();
            let char_start = master[..whole.start()].chars().count();
            let char_end = char_start + whole.as_str().chars().count();
            turns.push(MessageTurn {
                role: normalize_role(&role),
                content,
                char_start,
                char_end,
            });
        }
        if turns.is_empty() && !master.is_empty() {
            turns.push(MessageTurn {
                role: "user".into(),
                content: master.to_string(),
                char_start: 0,
                char_end: master.chars().count(),
            });
        }
        return turns;
    }

    // Plain / role-prefixed fallback.
    let mut turns = Vec::new();
    if !system_prompt.is_empty() {
        turns.push(MessageTurn {
            role: "system".into(),
            content: system_prompt.to_string(),
            char_start: 0,
            char_end: 0,
        });
    }
    if !master.is_empty() {
        turns.push(MessageTurn {
            role: "user".into(),
            content: master.to_string(),
            char_start: 0,
            char_end: master.chars().count(),
        });
    }
    turns
}

fn normalize_role(role: &str) -> String {
    match role.to_lowercase().as_str() {
        "human" | "user" => "user".into(),
        "assistant" | "model" | "bot" => "assistant".into(),
        "system" => "system".into(),
        "tool" | "function" => "tool".into(),
        other => other.to_string(),
    }
}

fn render_chatml(turns: &[MessageTurn], system_prompt: &str) -> String {
    let mut out = String::new();
    let mut wrote_system = false;
    if !system_prompt.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system_prompt);
        out.push_str("<|im_end|>\n");
        wrote_system = true;
    }
    for t in turns {
        let role = normalize_role(&t.role);
        if role == "system" && wrote_system {
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(&role);
        out.push('\n');
        out.push_str(&t.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn render_anthropic_blocks(turns: &[MessageTurn], system_prompt: &str) -> serde_json::Value {
    let mut system = system_prompt.to_string();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for t in turns {
        let role = normalize_role(&t.role);
        if role == "system" {
            if !system.is_empty() {
                system.push('\n');
            }
            system.push_str(&t.content);
            continue;
        }
        let api_role = if role == "assistant" { "assistant" } else { "user" };
        messages.push(serde_json::json!({
            "role": api_role,
            "content": [{"type": "text", "text": t.content}]
        }));
    }
    // Attach ephemeral cache_control to the last content block.
    if let Some(last) = messages.last_mut() {
        if let Some(content) = last.get_mut("content").and_then(|c| c.as_array_mut()) {
            if let Some(block) = content.last_mut() {
                block.as_object_mut().map(|o| {
                    o.insert(
                        "cache_control".into(),
                        serde_json::json!({"type": "ephemeral"}),
                    );
                });
            }
        }
    }
    serde_json::json!({
        "system": if system.is_empty() { serde_json::Value::Null } else { serde_json::json!(system) },
        "messages": messages
    })
}

fn render_openai_messages(turns: &[MessageTurn], system_prompt: &str) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system_prompt
        }));
    }
    for t in turns {
        let role = normalize_role(&t.role);
        if role == "system" && !system_prompt.is_empty() {
            continue;
        }
        messages.push(serde_json::json!({
            "role": role,
            "content": t.content
        }));
    }
    serde_json::json!({ "messages": messages })
}

fn render_llama3(turns: &[MessageTurn], system_prompt: &str) -> String {
    let mut out = String::from("<|begin_of_text|>");
    if !system_prompt.is_empty() {
        out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        out.push_str(system_prompt);
        out.push_str("<|eot_id|>");
    }
    for t in turns {
        let role = normalize_role(&t.role);
        if role == "system" && !system_prompt.is_empty() {
            continue;
        }
        out.push_str("<|start_header_id|>");
        out.push_str(&role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&t.content);
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn render_mistral(turns: &[MessageTurn], system_prompt: &str) -> String {
    let mut out = String::new();
    if !system_prompt.is_empty() {
        out.push_str(&format!("[INST] {}\n\n", system_prompt));
    }
    for t in turns {
        let role = normalize_role(&t.role);
        match role.as_str() {
            "user" => {
                if out.contains("[INST]") && !out.ends_with("[/INST]") {
                    out.push_str(&t.content);
                    out.push_str(" [/INST]");
                } else {
                    out.push_str("[INST] ");
                    out.push_str(&t.content);
                    out.push_str(" [/INST]");
                }
            }
            "assistant" => {
                out.push(' ');
                out.push_str(&t.content);
                out.push_str("</s>");
            }
            "system" => {}
            _ => {
                out.push_str(&t.content);
            }
        }
    }
    if !out.contains("[INST]") {
        out.insert_str(0, "[INST] ");
        out.push_str(" [/INST]");
    }
    out
}

fn strip_foreign_control_flags(text: &str, target: ModelFamily) -> String {
    let mut cleaned = text.to_string();
    match target {
        ModelFamily::OpenAI | ModelFamily::Grok => {
            cleaned = cleaned.replace("cache_control", "");
            cleaned = cleaned.replace("{\"type\": \"ephemeral\"}", "");
            cleaned = cleaned.replace("{\"type\":\"ephemeral\"}", "");
        }
        ModelFamily::Anthropic => {
            // Strip ChatML special tokens when targeting Anthropic blocks.
            cleaned = cleaned.replace("<|im_start|>", "");
            cleaned = cleaned.replace("<|im_end|>", "");
            cleaned = cleaned.replace("<|endoftext|>", "");
        }
        ModelFamily::OpenWeight => {
            cleaned = cleaned.replace("cache_control", "");
            cleaned = cleaned.replace("{\"type\": \"ephemeral\"}", "");
            cleaned = cleaned.replace("{\"type\":\"ephemeral\"}", "");
        }
    }
    cleaned
}

impl SessionEngine {
    pub fn new(system_prompt: Option<String>) -> UcpResult<Self> {
        let mut state = SessionState::new()?;
        if let Some(sp) = system_prompt {
            state.system_prompt = sp;
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(state)),
        })
    }

    /// Append text to the immutable-linear master context and realign all tokenizers.
    pub fn append_text_and_align(&self, text: String) -> UcpResult<(usize, usize, usize)> {
        if text.is_empty() {
            return Ok(self.token_counts());
        }
        let mut state = self.inner.write();
        state.master_context.push_str(&text);
        let master_chars = state.master_context.chars().count();
        if text.contains("Human:")
            || text.contains("Assistant:")
            || text.contains("<|im_start|>")
            || text.contains("User:")
        {
            state.turns = parse_turns_from_master(&state.master_context, &state.system_prompt);
        } else if let Some(last) = state.turns.last_mut() {
            last.content.push_str(&text);
            last.char_end = master_chars;
        } else {
            let start = master_chars - text.chars().count();
            state.turns.push(MessageTurn {
                role: "user".into(),
                content: text.clone(),
                char_start: start,
                char_end: master_chars,
            });
        }
        realign_all(&mut state).map_err(UcpError::Tokenize)?;
        Ok((
            state.openai_table.token_count(),
            state.anthropic_table.token_count(),
            state.openweight_table.token_count(),
        ))
    }

    /// Append a structured role turn (preferred over raw text for chat).
    pub fn append_turn(&self, role: String, content: String) -> UcpResult<(usize, usize, usize)> {
        let role_n = normalize_role(&role);
        let formatted = match role_n.as_str() {
            "system" => format!("System: {}\n", content),
            "user" => format!("Human: {}\n", content),
            "assistant" => format!("Assistant: {}\n", content),
            "tool" => format!("Tool: {}\n", content),
            other => format!("{}: {}\n", other, content),
        };
        let mut state = self.inner.write();
        let char_start = state.master_context.chars().count();
        state.master_context.push_str(&formatted);
        let char_end = state.master_context.chars().count();
        state.turns.push(MessageTurn {
            role: role_n,
            content,
            char_start,
            char_end,
        });
        realign_all(&mut state).map_err(UcpError::Tokenize)?;
        Ok((
            state.openai_table.token_count(),
            state.anthropic_table.token_count(),
            state.openweight_table.token_count(),
        ))
    }

    /// OpenAI Chat Completions JSON payload.
    pub fn get_openai_payload(&self) -> UcpResult<String> {
        let state = self.inner.read();
        let cleaned = strip_foreign_control_flags(&state.master_context, ModelFamily::OpenAI);
        let turns = if state.turns.is_empty() {
            parse_turns_from_master(&cleaned, &state.system_prompt)
        } else {
            state.turns.clone()
        };
        let payload = render_openai_messages(&turns, &state.system_prompt);
        let mut obj = payload.as_object().cloned().unwrap_or_default();
        obj.insert("stream".into(), serde_json::json!(true));
        obj.remove("cache_control");
        serde_json::to_string(&obj).map_err(|e| UcpError::Serialize(e.to_string()))
    }

    /// Anthropic Messages API JSON payload with ephemeral cache_control on last block.
    pub fn get_anthropic_payload(&self) -> UcpResult<String> {
        let state = self.inner.read();
        let cleaned = strip_foreign_control_flags(&state.master_context, ModelFamily::Anthropic);
        let turns = if state.turns.is_empty() {
            parse_turns_from_master(&cleaned, &state.system_prompt)
        } else {
            state.turns.clone()
        };
        let mut payload = render_anthropic_blocks(&turns, &state.system_prompt);
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("stream".into(), serde_json::json!(true));
            if let Some(sys) = obj.get_mut("system") {
                if let Some(s) = sys.as_str() {
                    *sys = serde_json::json!(s
                        .replace("<|im_start|>", "")
                        .replace("<|im_end|>", ""));
                }
            }
        }
        serde_json::to_string(&payload).map_err(|e| UcpError::Serialize(e.to_string()))
    }

    /// Grok (xAI) uses OpenAI-compatible messages.
    pub fn get_grok_payload(&self) -> UcpResult<String> {
        self.get_openai_payload()
    }

    /// Open-weight / local OpenAI-compatible payload.
    pub fn get_openweight_payload(&self, include_raw_prompt: bool) -> UcpResult<String> {
        let state = self.inner.read();
        let cleaned = strip_foreign_control_flags(&state.master_context, ModelFamily::OpenWeight);
        let turns = if state.turns.is_empty() {
            parse_turns_from_master(&cleaned, &state.system_prompt)
        } else {
            state.turns.clone()
        };
        let mut payload = render_openai_messages(&turns, &state.system_prompt);
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("stream".into(), serde_json::json!(true));
            if include_raw_prompt {
                let raw = match state.openweight_template {
                    PromptTemplate::ChatML => render_chatml(&turns, &state.system_prompt),
                    PromptTemplate::Llama3 => render_llama3(&turns, &state.system_prompt),
                    PromptTemplate::MistralInstruct => render_mistral(&turns, &state.system_prompt),
                    PromptTemplate::AnthropicBlocks => render_chatml(&turns, &state.system_prompt),
                    PromptTemplate::Plain => cleaned,
                };
                obj.insert("raw_prompt".into(), serde_json::json!(raw));
                obj.insert(
                    "template".into(),
                    serde_json::json!(format!("{:?}", state.openweight_template)),
                );
            }
        }
        serde_json::to_string(&payload).map_err(|e| UcpError::Serialize(e.to_string()))
    }

    /// Hot-swap active family + template without rewriting master_context.
    pub fn swap_model(
        &self,
        family: &str,
        template: Option<&str>,
    ) -> UcpResult<(String, String)> {
        let mut state = self.inner.write();
        let fam = parse_family(family)?;
        let tmpl = if let Some(t) = template {
            parse_template(t)?
        } else {
            default_template_for(fam)
        };
        state.active_family = fam;
        state.active_template = tmpl;
        if fam == ModelFamily::OpenWeight {
            state.openweight_template = tmpl;
        }
        Ok((format!("{:?}", fam), format!("{:?}", tmpl)))
    }

    pub fn get_active_model(&self) -> String {
        format!("{:?}", self.inner.read().active_family)
    }

    pub fn get_master_context(&self) -> String {
        self.inner.read().master_context.clone()
    }

    pub fn get_system_prompt(&self) -> String {
        self.inner.read().system_prompt.clone()
    }

    pub fn set_system_prompt(&self, prompt: String) {
        self.inner.write().system_prompt = prompt;
    }

    /// Token counts: (openai, anthropic, openweight).
    pub fn token_counts(&self) -> (usize, usize, usize) {
        let state = self.inner.read();
        (
            state.openai_table.token_count(),
            state.anthropic_table.token_count(),
            state.openweight_table.token_count(),
        )
    }

    /// Cache warmth factor for a provider in \[0.0, 1.0\].
    pub fn cache_warmth_factor(&self, provider: &str) -> f64 {
        let state = self.inner.read();
        let key = provider.to_lowercase();
        let warm = *state.cache_warm_tokens.get(&key).unwrap_or(&0);
        let live = match key.as_str() {
            "openai" | "gpt" | "grok" => state.openai_table.token_count(),
            "anthropic" | "claude" => state.anthropic_table.token_count(),
            _ => state.openweight_table.token_count(),
        };
        if live == 0 {
            return 0.0;
        }
        let covered = warm.min(live) as f64;
        let denom = live.max(CACHE_THRESHOLD_TOKENS) as f64;
        (covered / denom).clamp(0.0, 1.0)
    }

    pub fn set_cache_warm_tokens(&self, provider: &str, tokens: usize) {
        let mut state = self.inner.write();
        state
            .cache_warm_tokens
            .insert(provider.to_lowercase(), tokens);
    }

    /// Load a HuggingFace `tokenizer.json` for Anthropic-style or open-weight alignment.
    pub fn load_hf_tokenizer(&self, path: String, target: &str) -> UcpResult<String> {
        let p = Path::new(&path);
        if !p.exists() {
            return Err(UcpError::TokenizerNotFound(format!(
                "tokenizer file not found: {path}"
            )));
        }
        let tok = Tokenizer::from_file(p)
            .map_err(|e| UcpError::Tokenize(format!("failed to load tokenizer: {e}")))?;
        let backend = TokenizerBackend {
            name: path.clone(),
            kind: BackendKind::HuggingFace(Arc::new(tok)),
        };
        let mut state = self.inner.write();
        match target.to_lowercase().as_str() {
            "anthropic" | "claude" => {
                state.anthropic_backend = backend;
                state.anthropic_table.clear();
            }
            "openweight" | "hf" | "local" => {
                state.openweight_backend = backend;
                state.openweight_table.clear();
            }
            alias => {
                state.extra_backends.insert(alias.to_string(), backend);
                state
                    .extra_tables
                    .insert(alias.to_string(), AlignmentTable::default());
            }
        }
        realign_all(&mut state).map_err(UcpError::Tokenize)?;
        Ok(format!("loaded {path} for {target}"))
    }

    /// Export alignment table for a family.
    pub fn get_alignment_table(&self, family: &str) -> UcpResult<Vec<TokenSpan>> {
        let state = self.inner.read();
        let spans = match family.to_lowercase().as_str() {
            "openai" | "gpt" | "grok" => state.openai_table.spans.clone(),
            "anthropic" | "claude" => state.anthropic_table.spans.clone(),
            "openweight" | "hf" | "local" => state.openweight_table.spans.clone(),
            other => {
                if let Some(t) = state.extra_tables.get(other) {
                    t.spans.clone()
                } else {
                    return Err(UcpError::InvalidFamily(format!("unknown family: {other}")));
                }
            }
        };
        Ok(spans)
    }

    /// Map an absolute character index to the nearest token index for a family.
    pub fn char_to_token_index(&self, family: &str, char_index: usize) -> usize {
        let state = self.inner.read();
        let spans: &[TokenSpan] = match family.to_lowercase().as_str() {
            "openai" | "gpt" | "grok" => &state.openai_table.spans,
            "anthropic" | "claude" => &state.anthropic_table.spans,
            _ => &state.openweight_table.spans,
        };
        if spans.is_empty() {
            return 0;
        }
        for (i, s) in spans.iter().enumerate() {
            if char_index >= s.char_start && char_index < s.char_end {
                return i;
            }
        }
        spans.len().saturating_sub(1)
    }

    /// Snapshot of engine stats as JSON-friendly map values.
    pub fn stats(&self) -> HashMap<String, serde_json::Value> {
        let state = self.inner.read();
        let mut d = HashMap::new();
        d.insert(
            "master_chars".into(),
            serde_json::json!(state.master_context.chars().count()),
        );
        d.insert(
            "openai_tokens".into(),
            serde_json::json!(state.openai_table.token_count()),
        );
        d.insert(
            "anthropic_tokens".into(),
            serde_json::json!(state.anthropic_table.token_count()),
        );
        d.insert(
            "openweight_tokens".into(),
            serde_json::json!(state.openweight_table.token_count()),
        );
        d.insert(
            "active_family".into(),
            serde_json::json!(format!("{:?}", state.active_family)),
        );
        d.insert(
            "active_template".into(),
            serde_json::json!(format!("{:?}", state.active_template)),
        );
        d.insert("turns".into(), serde_json::json!(state.turns.len()));
        d.insert(
            "cache_warmth_openai".into(),
            serde_json::json!(self.cache_warmth_factor("openai")),
        );
        d.insert(
            "cache_warmth_anthropic".into(),
            serde_json::json!(self.cache_warmth_factor("anthropic")),
        );
        d.insert(
            "cache_warmth_openweight".into(),
            serde_json::json!(self.cache_warmth_factor("openweight")),
        );
        d.insert(
            "cache_threshold".into(),
            serde_json::json!(CACHE_THRESHOLD_TOKENS),
        );
        d
    }

    pub fn clear(&self) {
        let mut state = self.inner.write();
        state.master_context.clear();
        state.turns.clear();
        state.openai_table.clear();
        state.anthropic_table.clear();
        state.openweight_table.clear();
        for t in state.extra_tables.values_mut() {
            t.clear();
        }
        for v in state.cache_warm_tokens.values_mut() {
            *v = 0;
        }
    }
}

pub fn parse_family(s: &str) -> UcpResult<ModelFamily> {
    match s.to_lowercase().as_str() {
        "openai" | "gpt" | "gpt-5.5" | "gpt5.5" | "oai" => Ok(ModelFamily::OpenAI),
        "anthropic" | "claude" | "claude-3.5" | "sonnet" => Ok(ModelFamily::Anthropic),
        "grok" | "xai" => Ok(ModelFamily::Grok),
        "openweight" | "open" | "local" | "hf" | "ollama" | "vllm" => Ok(ModelFamily::OpenWeight),
        other => Err(UcpError::InvalidFamily(format!("unknown model family: {other}"))),
    }
}

pub fn parse_template(s: &str) -> UcpResult<PromptTemplate> {
    match s.to_lowercase().as_str() {
        "chatml" => Ok(PromptTemplate::ChatML),
        "anthropic" | "anthropic_blocks" | "blocks" => Ok(PromptTemplate::AnthropicBlocks),
        "llama3" | "llama" => Ok(PromptTemplate::Llama3),
        "mistral" | "mistral_instruct" => Ok(PromptTemplate::MistralInstruct),
        "plain" => Ok(PromptTemplate::Plain),
        other => Err(UcpError::InvalidTemplate(format!("unknown template: {other}"))),
    }
}

pub fn default_template_for(fam: ModelFamily) -> PromptTemplate {
    match fam {
        ModelFamily::OpenAI | ModelFamily::Grok => PromptTemplate::ChatML,
        ModelFamily::Anthropic => PromptTemplate::AnthropicBlocks,
        ModelFamily::OpenWeight => PromptTemplate::ChatML,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_tokenize_tracks_offsets() {
        let text = "Hello, world!";
        let spans = approx_tokenize(text, 0);
        assert!(!spans.is_empty());
        assert_eq!(spans.first().unwrap().char_start, 0);
        assert_eq!(spans.last().unwrap().char_end, text.chars().count());
    }

    #[test]
    fn chatml_render_contains_markers() {
        let turns = vec![MessageTurn {
            role: "user".into(),
            content: "hi".into(),
            char_start: 0,
            char_end: 2,
        }];
        let s = render_chatml(&turns, "sys");
        assert!(s.contains("<|im_start|>system"));
        assert!(s.contains("<|im_start|>user"));
        assert!(s.contains("<|im_start|>assistant"));
    }

    #[test]
    fn anthropic_payload_has_cache_control() {
        let turns = vec![
            MessageTurn {
                role: "user".into(),
                content: "hello".into(),
                char_start: 0,
                char_end: 5,
            },
            MessageTurn {
                role: "assistant".into(),
                content: "world".into(),
                char_start: 5,
                char_end: 10,
            },
        ];
        let v = render_anthropic_blocks(&turns, "sys");
        let msgs = v["messages"].as_array().unwrap();
        let last = msgs.last().unwrap();
        let block = &last["content"][0];
        assert_eq!(block["cache_control"]["type"], "ephemeral");
    }
}
