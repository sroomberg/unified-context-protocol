"""Configuration for the hot-swap REPL harness."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal[
    "openai",
    "anthropic",
    "grok",
    "openweight",
]


class Settings(BaseSettings):
    """Environment-driven settings for provider gateways and the TUI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Active provider at session start.
    default_provider: ProviderName = "anthropic"

    # --- OpenAI ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    openai_model: str = Field(default="gpt-5.5", alias="OPENAI_MODEL")

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        alias="ANTHROPIC_BASE_URL",
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        alias="ANTHROPIC_MODEL",
    )
    anthropic_version: str = Field(default="2023-06-01", alias="ANTHROPIC_VERSION")

    # --- Grok / xAI (OpenAI-compatible) ---
    grok_api_key: str = Field(default="", alias="XAI_API_KEY")
    grok_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")
    grok_model: str = Field(default="grok-3", alias="XAI_MODEL")

    # --- Open-weight / local OpenAI-compatible (vLLM, Ollama, LM Studio, Together, …) ---
    openweight_api_key: str = Field(default="", alias="OPENWEIGHT_API_KEY")
    openweight_base_url: str = Field(
        default="http://127.0.0.1:11434/v1",
        alias="OPENWEIGHT_BASE_URL",
    )
    openweight_model: str = Field(default="llama3.1", alias="OPENWEIGHT_MODEL")
    openweight_template: str = Field(default="chatml", alias="OPENWEIGHT_TEMPLATE")
    openweight_tokenizer_path: str = Field(default="", alias="OPENWEIGHT_TOKENIZER_PATH")
    # Enable vLLM / SGLang automatic prefix caching headers when supported.
    openweight_enable_prefix_cache: bool = Field(
        default=True,
        alias="OPENWEIGHT_ENABLE_PREFIX_CACHE",
    )

    # Optional HF tokenizer for Anthropic-side alignment.
    anthropic_tokenizer_path: str = Field(default="", alias="ANTHROPIC_TOKENIZER_PATH")

    # Session
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant in a cross-model hot-swap REPL. "
            "Prior turns may have been produced by a different model family; "
            "continue coherently without mentioning the swap unless asked."
        ),
        alias="SYSTEM_PROMPT",
    )
    max_output_tokens: int = Field(default=4096, alias="MAX_OUTPUT_TOKENS")
    request_timeout_s: float = Field(default=120.0, alias="REQUEST_TIMEOUT_S")
    dual_write_enabled: bool = Field(default=True, alias="DUAL_WRITE_ENABLED")
    cache_threshold_tokens: int = Field(default=1024, alias="CACHE_THRESHOLD_TOKENS")

    @field_validator("default_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            v = value.strip().lower()
            aliases = {
                "gpt": "openai",
                "claude": "anthropic",
                "sonnet": "anthropic",
                "xai": "grok",
                "local": "openweight",
                "ollama": "openweight",
                "vllm": "openweight",
                "hf": "openweight",
                "open": "openweight",
            }
            return aliases.get(v, v)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
