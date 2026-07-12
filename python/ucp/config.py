"""Protocol-level settings (no API keys, endpoints, or runtime knobs)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal["openai", "anthropic", "grok", "openweight"]
TemplateName = Literal["chatml", "anthropic_blocks", "llama3", "mistral", "plain"]


class ProtocolSettings(BaseSettings):
    """UCP infra configuration: templates, thresholds, optional tokenizer paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        env_prefix="UCP_",
    )

    default_family: ProviderName = Field(default="anthropic")
    openweight_template: TemplateName = Field(default="chatml")
    anthropic_tokenizer_path: str = Field(default="")
    openweight_tokenizer_path: str = Field(default="")
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Prior turns may have been produced by "
            "a different model family; continue coherently."
        )
    )
    cache_threshold_tokens: int = Field(default=1024)
    openweight_enable_prefix_cache_hint: bool = Field(default=True)

    @field_validator("default_family", mode="before")
    @classmethod
    def _normalize_family(cls, value: object) -> object:
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

    @field_validator("openweight_template", mode="before")
    @classmethod
    def _normalize_template(cls, value: object) -> object:
        if isinstance(value, str):
            v = value.strip().lower().replace("-", "_")
            aliases = {
                "anthropic": "anthropic_blocks",
                "blocks": "anthropic_blocks",
                "llama": "llama3",
                "mistral_instruct": "mistral",
            }
            return aliases.get(v, v)
        return value


@lru_cache(maxsize=1)
def get_settings() -> ProtocolSettings:
    """Return a process-wide cached ProtocolSettings instance."""
    return ProtocolSettings()
