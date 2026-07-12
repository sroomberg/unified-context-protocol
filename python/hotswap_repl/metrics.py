"""Session metrics helpers (cache warmth, token deltas)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hotswap_repl.providers import ProviderId


@dataclass(frozen=True)
class SessionMetrics:
    active: ProviderId
    openai_tokens: int
    anthropic_tokens: int
    openweight_tokens: int
    warmth_openai: float
    warmth_anthropic: float
    warmth_grok: float
    warmth_openweight: float
    master_chars: int
    turns: int
    cache_threshold: int

    @property
    def dormant_warmth(self) -> float:
        """Warmth of the best non-active proprietary/open-weight peer."""
        mapping = {
            ProviderId.OPENAI: self.warmth_openai,
            ProviderId.ANTHROPIC: self.warmth_anthropic,
            ProviderId.GROK: self.warmth_grok,
            ProviderId.OPENWEIGHT: self.warmth_openweight,
        }
        peers = [v for k, v in mapping.items() if k is not self.active]
        return max(peers) if peers else 0.0

    def warmth_for(self, provider: ProviderId) -> float:
        return {
            ProviderId.OPENAI: self.warmth_openai,
            ProviderId.ANTHROPIC: self.warmth_anthropic,
            ProviderId.GROK: self.warmth_grok,
            ProviderId.OPENWEIGHT: self.warmth_openweight,
        }[provider]


def collect_metrics(engine: Any, active: ProviderId) -> SessionMetrics:
    stats = engine.stats()
    return SessionMetrics(
        active=active,
        openai_tokens=int(stats.get("openai_tokens", 0)),
        anthropic_tokens=int(stats.get("anthropic_tokens", 0)),
        openweight_tokens=int(stats.get("openweight_tokens", 0)),
        warmth_openai=float(engine.cache_warmth_factor("openai")),
        warmth_anthropic=float(engine.cache_warmth_factor("anthropic")),
        warmth_grok=float(engine.cache_warmth_factor("grok")),
        warmth_openweight=float(engine.cache_warmth_factor("openweight")),
        master_chars=int(stats.get("master_chars", 0)),
        turns=int(stats.get("turns", 0)),
        cache_threshold=int(stats.get("cache_threshold", 1024)),
    )


def format_warmth(value: float) -> str:
    pct = max(0.0, min(1.0, value)) * 100.0
    bar_len = 10
    filled = int(round(pct / 100.0 * bar_len))
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"{bar} {pct:5.1f}%"
