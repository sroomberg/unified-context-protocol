"""Unified Context Protocol (UCP) infrastructure.

Pure context-alignment and payload-transposition layer. No network I/O,
no TUI, no agent runtime — those belong in separate clients (e.g. OpenCode).
"""

from __future__ import annotations

__version__ = "0.1.0"

try:
    from ucp._engine import (
        CACHE_THRESHOLD_TOKENS,
        ModelFamily,
        PromptTemplate,
        SessionEngine,
    )
except ImportError:  # pragma: no cover - allows imports before maturin build
    CACHE_THRESHOLD_TOKENS = 1024
    ModelFamily = None  # type: ignore[assignment,misc]
    PromptTemplate = None  # type: ignore[assignment,misc]
    SessionEngine = None  # type: ignore[assignment,misc]

from ucp.config import ProtocolSettings, get_settings
from ucp.providers import ProviderId, build_request_body, ensure_anthropic_cache_control

__all__ = [
    "CACHE_THRESHOLD_TOKENS",
    "ModelFamily",
    "PromptTemplate",
    "ProtocolSettings",
    "ProviderId",
    "SessionEngine",
    "build_request_body",
    "ensure_anthropic_cache_control",
    "get_settings",
    "__version__",
]
