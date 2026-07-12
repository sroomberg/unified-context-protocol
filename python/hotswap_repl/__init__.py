"""Cross-model hot-swapping REPL harness.

Python orchestration layer over the Rust ``SessionEngine`` core, with async
dual-write prefix warming for OpenAI, Anthropic, Grok, and open-weight providers.
"""

from __future__ import annotations

__version__ = "0.1.0"

try:
    from hotswap_repl._engine import (
        CACHE_THRESHOLD_TOKENS,
        ModelFamily,
        PromptTemplate,
        SessionEngine,
    )
except ImportError:  # pragma: no cover - allows docs/imports before maturin build
    CACHE_THRESHOLD_TOKENS = 1024
    ModelFamily = None  # type: ignore[assignment,misc]
    PromptTemplate = None  # type: ignore[assignment,misc]
    SessionEngine = None  # type: ignore[assignment,misc]

from hotswap_repl.config import Settings, get_settings
from hotswap_repl.providers import ProviderId, ProviderRegistry

__all__ = [
    "CACHE_THRESHOLD_TOKENS",
    "ModelFamily",
    "PromptTemplate",
    "ProviderId",
    "ProviderRegistry",
    "SessionEngine",
    "Settings",
    "get_settings",
    "__version__",
]
