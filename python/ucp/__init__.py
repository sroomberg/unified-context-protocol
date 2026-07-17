"""Optional Python bindings for Unified Context Protocol (UCP).

The protocol core is the Rust ``ucp`` crate. This package is a convenience
layer (PyO3 ``SessionEngine`` plus pure payload helpers). Runtimes that do
not need Python can depend on the Rust crate alone.
"""

from __future__ import annotations

__version__ = "0.1.0"  # x-release-please-version

try:
    from ucp._engine import (
        CACHE_THRESHOLD_TOKENS,
        VERSION as _ENGINE_VERSION,
        ModelFamily,
        PromptTemplate,
        SessionEngine,
    )
except ImportError:  # pragma: no cover - allows imports before maturin build
    CACHE_THRESHOLD_TOKENS = 1024
    _ENGINE_VERSION = None  # type: ignore[assignment]
    ModelFamily = None  # type: ignore[assignment,misc]
    PromptTemplate = None  # type: ignore[assignment,misc]
    SessionEngine = None  # type: ignore[assignment,misc]
else:
    # Prefer the compiled crate version when the extension is present.
    if _ENGINE_VERSION:
        __version__ = _ENGINE_VERSION

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
