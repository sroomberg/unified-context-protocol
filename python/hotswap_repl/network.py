"""Async network pipeline: dual-write warmup + active-provider streaming."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import httpx

from hotswap_repl.config import Settings, get_settings
from hotswap_repl.providers import (
    ProviderId,
    ProviderRegistry,
    build_request_body,
    iter_sse_text,
)

logger = logging.getLogger(__name__)


@dataclass
class WarmupResult:
    provider: ProviderId
    ok: bool
    tokens_hint: int = 0
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class StreamChunk:
    text: str
    provider: ProviderId
    done: bool = False


@dataclass
class NetworkPipeline:
    """Coordinates non-blocking calls to model gateways with speculative dual-write."""

    engine: Any
    settings: Settings = field(default_factory=get_settings)
    registry: ProviderRegistry | None = None
    active: ProviderId = ProviderId.ANTHROPIC
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _warmup_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _last_warmup: dict[ProviderId, WarmupResult] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = ProviderRegistry(self.settings)
        try:
            self.active = ProviderId.parse(self.settings.default_provider)
        except ValueError:
            self.active = ProviderId.ANTHROPIC

    async def __aenter__(self) -> "NetworkPipeline":
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_s),
                follow_redirects=True,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("NetworkPipeline is not open; call open() or use async with")
        return self._client

    def set_active(self, provider: ProviderId | str) -> ProviderId:
        pid = provider if isinstance(provider, ProviderId) else ProviderId.parse(provider)
        self.active = pid
        family = pid.engine_family
        template = None
        if pid is ProviderId.OPENWEIGHT:
            template = self.settings.openweight_template
        self.engine.swap_model(family, template)
        return pid

    async def dual_write_warmup(
        self,
        text: str | None = None,
        *,
        providers: list[ProviderId] | None = None,
        max_tokens: int = 1,
    ) -> list[WarmupResult]:
        """Speculatively broadcast the current prefix to all configured providers.

        Sends a minimal completion request (``max_tokens=1``) so upstream prompt
        caches materialize for the shared linear history. Safe to call after
        appending long system docs or after each user turn.
        """
        if not self.settings.dual_write_enabled:
            return []

        async with self._warmup_lock:
            if text:
                # Optional explicit document priming (does not append to master).
                logger.debug("dual_write priming document chars=%s", len(text))

            targets = providers or self.registry.configured_for_warmup()  # type: ignore[union-attr]
            if not targets:
                return []

            client = self._require_client()
            tasks = [
                self._warmup_one(client, pid, max_tokens=max_tokens) for pid in targets
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out: list[WarmupResult] = []
            for pid, result in zip(targets, results):
                if isinstance(result, Exception):
                    wr = WarmupResult(provider=pid, ok=False, error=str(result))
                else:
                    wr = result
                out.append(wr)
                self._last_warmup[pid] = wr
                if wr.ok:
                    self.engine.set_cache_warm_tokens(pid.value, wr.tokens_hint)
            return out

    async def _warmup_one(
        self,
        client: httpx.AsyncClient,
        provider: ProviderId,
        *,
        max_tokens: int,
    ) -> WarmupResult:
        assert self.registry is not None
        endpoint = self.registry.get(provider)
        body = build_request_body(
            provider,
            self.engine,
            self.settings,
            stream=False,
            max_tokens=max_tokens,
        )
        # Warmup must not stream; drop SSE preference.
        headers = endpoint.auth_headers()
        headers["Accept"] = "application/json"

        oai, ant, ow = self.engine.token_counts()
        tokens_hint = {
            ProviderId.OPENAI: oai,
            ProviderId.GROK: oai,
            ProviderId.ANTHROPIC: ant,
            ProviderId.OPENWEIGHT: ow,
        }[provider]

        started = asyncio.get_running_loop().time()
        try:
            response = await client.post(
                endpoint.chat_url,
                headers=headers,
                json=body,
            )
            latency = (asyncio.get_running_loop().time() - started) * 1000.0
            if response.status_code >= 400:
                # Local open-weight servers may be offline; treat as soft failure.
                detail = response.text[:400]
                return WarmupResult(
                    provider=provider,
                    ok=False,
                    tokens_hint=tokens_hint,
                    error=f"HTTP {response.status_code}: {detail}",
                    latency_ms=latency,
                )
            # Prefer usage.prompt_tokens when present.
            try:
                payload = response.json()
                usage = payload.get("usage") or {}
                prompt_tokens = (
                    usage.get("prompt_tokens")
                    or usage.get("input_tokens")
                    or tokens_hint
                )
                tokens_hint = int(prompt_tokens)
            except Exception:  # noqa: BLE001
                pass
            return WarmupResult(
                provider=provider,
                ok=True,
                tokens_hint=tokens_hint,
                latency_ms=latency,
            )
        except httpx.HTTPError as exc:
            latency = (asyncio.get_running_loop().time() - started) * 1000.0
            return WarmupResult(
                provider=provider,
                ok=False,
                tokens_hint=tokens_hint,
                error=str(exc),
                latency_ms=latency,
            )

    async def stream_completion(
        self,
        *,
        provider: ProviderId | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream tokens from the active (or specified) provider."""
        pid = provider or self.active
        assert self.registry is not None
        endpoint = self.registry.get(pid)
        body = build_request_body(
            pid,
            self.engine,
            self.settings,
            stream=True,
        )
        client = self._require_client()
        headers = endpoint.auth_headers()

        try:
            async with client.stream(
                "POST",
                endpoint.chat_url,
                headers=headers,
                json=body,
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:800]
                    msg = f"{pid.value} HTTP {response.status_code}: {detail}"
                    if on_error:
                        on_error(msg)
                    raise RuntimeError(msg)
                async for text in iter_sse_text(response, pid):
                    yield StreamChunk(text=text, provider=pid)
                yield StreamChunk(text="", provider=pid, done=True)
        except httpx.HTTPError as exc:
            msg = f"{pid.value} network error: {exc}"
            if on_error:
                on_error(msg)
            raise RuntimeError(msg) from exc

    async def chat_turn(
        self,
        user_text: str,
        *,
        warmup: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """Append a user turn, optionally dual-write warm, then stream the reply."""
        self.engine.append_turn("user", user_text)
        if warmup and self.settings.dual_write_enabled:
            # Fire-and-forget warmup for dormant providers; do not block TTFT hard.
            asyncio.create_task(self._safe_warmup())
        assistant_buf: list[str] = []
        try:
            async for chunk in self.stream_completion():
                if chunk.text:
                    assistant_buf.append(chunk.text)
                yield chunk
        finally:
            full = "".join(assistant_buf)
            if full:
                self.engine.append_turn("assistant", full)
                if warmup and self.settings.dual_write_enabled:
                    asyncio.create_task(self._safe_warmup())

    async def _safe_warmup(self) -> None:
        try:
            await self.dual_write_warmup()
        except Exception:  # noqa: BLE001
            logger.exception("dual_write_warmup failed")

    def cache_warmth(self, provider: ProviderId | str) -> float:
        pid = provider if isinstance(provider, ProviderId) else ProviderId.parse(provider)
        return float(self.engine.cache_warmth_factor(pid.value))

    def token_counts(self) -> tuple[int, int, int]:
        return tuple(self.engine.token_counts())  # type: ignore[return-value]

    def last_warmup(self, provider: ProviderId) -> WarmupResult | None:
        return self._last_warmup.get(provider)
