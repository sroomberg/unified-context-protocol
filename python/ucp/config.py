"""Protocol-level settings (no API keys, endpoints, or runtime knobs)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal["openai", "anthropic", "grok", "openweight"]
TemplateName = Literal["chatml", "anthropic_blocks", "llama3", "mistral", "plain"]


class ProtocolSettings(BaseSettings):
    """UCP infra configuration: templates, thresholds, optional tokenizer paths.

    All fields can be overridden via ``UCP_*`` environment variables (see
    ``.env.example``). This settings object never holds API keys or base URLs —
    those belong to the consuming runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        env_prefix="UCP_",
    )

    default_family: ProviderName = Field(
        default="anthropic",
        description="Default model family alias for new sessions.",
    )
    openweight_template: TemplateName = Field(
        default="chatml",
        description="Prompt template used when rendering open-weight raw prompts.",
    )
    anthropic_tokenizer_path: str = Field(
        default="",
        description="Optional path to a HuggingFace tokenizer.json for Anthropic alignment.",
    )
    openweight_tokenizer_path: str = Field(
        default="",
        description="Optional path to a HuggingFace tokenizer.json for open-weight alignment.",
    )
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Prior turns may have been produced by "
            "a different model family; continue coherently."
        ),
        description="Default system prompt seeded into SessionEngine.",
    )
    cache_threshold_tokens: int = Field(
        default=1024,
        description="Token threshold typically required before provider caches activate.",
    )
    openweight_enable_prefix_cache_hint: bool = Field(
        default=True,
        description="When true, open-weight payloads include a prefix_caching hint.",
    )

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
