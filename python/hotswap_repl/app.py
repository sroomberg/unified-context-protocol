"""Textual TUI for the cross-model hot-swapping REPL."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Optional

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

from hotswap_repl.config import Settings, get_settings
from hotswap_repl.metrics import collect_metrics, format_warmth
from hotswap_repl.network import NetworkPipeline
from hotswap_repl.providers import ProviderId

logger = logging.getLogger(__name__)

HELP_TEXT = """\
[b]Commands[/b]
  /swap gpt|claude|grok|open   Hot-swap active provider
  /warmup                      Dual-write warm both caches now
  /stats                       Print engine alignment stats
  /clear                       Clear master context (new session)
  /help                        Show this help
  /quit                        Exit

Type normally to chat. Prefixes stay warm on dormant models.
"""


class SwapBlink(Message):
    """Posted when a hot-swap completes so the UI can flash confirmation."""

    def __init__(self, provider: ProviderId) -> None:
        super().__init__()
        self.provider = provider


class StatusSidebar(Static):
    """Fixed sidebar: active model, token counts, dormant cache warmth."""

    active_label: reactive[str] = reactive("—")
    openai_tokens: reactive[int] = reactive(0)
    anthropic_tokens: reactive[int] = reactive(0)
    openweight_tokens: reactive[int] = reactive(0)
    warmth_dormant: reactive[str] = reactive("░░░░░░░░░░  0.0%")
    warmth_detail: reactive[str] = reactive("")
    swap_flash: reactive[bool] = reactive(False)

    def render(self) -> Text:
        title = Text()
        if self.swap_flash:
            title.append("⚡ HOT-SWAP\n", style="bold yellow on dark_green")
        else:
            title.append("Session Status\n", style="bold cyan")
        title.append("──────────────\n", style="dim")
        title.append("Active\n", style="bold")
        title.append(f"  {self.active_label}\n\n", style="bright_white")
        title.append("Token Counts\n", style="bold")
        title.append(f"  OpenAI   {self.openai_tokens:>7}\n", style="green")
        title.append(f"  Anthropic{self.anthropic_tokens:>7}\n", style="magenta")
        title.append(f"  OpenWeight{self.openweight_tokens:>6}\n\n", style="blue")
        title.append("Dormant Cache Warmth\n", style="bold")
        title.append(f"  {self.warmth_dormant}\n", style="bright_yellow")
        if self.warmth_detail:
            title.append(f"\n{self.warmth_detail}\n", style="dim")
        title.append("\n/swap gpt|claude|grok|open\n", style="dim")
        return title

    def update_from_engine(self, engine: Any, active: ProviderId) -> None:
        metrics = collect_metrics(engine, active)
        self.active_label = active.display_name
        self.openai_tokens = metrics.openai_tokens
        self.anthropic_tokens = metrics.anthropic_tokens
        self.openweight_tokens = metrics.openweight_tokens
        self.warmth_dormant = format_warmth(metrics.dormant_warmth)
        lines = [
            f"  gpt   {format_warmth(metrics.warmth_openai)}",
            f"  claude{format_warmth(metrics.warmth_anthropic)}",
            f"  grok  {format_warmth(metrics.warmth_grok)}",
            f"  open  {format_warmth(metrics.warmth_openweight)}",
            f"\n  chars {metrics.master_chars}  turns {metrics.turns}",
            f"  threshold ≥{metrics.cache_threshold} tok",
        ]
        self.warmth_detail = "\n".join(lines)


class ChatFeed(RichLog):
    """Scrolling chat feed that accepts streamed token fragments."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(wrap=True, highlight=True, markup=True, **kwargs)
        self._stream_buffer = Text()
        self._streaming = False

    def add_user(self, text: str) -> None:
        self.write(Text.from_markup(f"\n[bold cyan]You[/]  {text}"))

    def add_system(self, text: str, style: str = "dim") -> None:
        self.write(Text(f"\n{text}", style=style))

    def begin_assistant(self, label: str) -> None:
        self._streaming = True
        self._stream_buffer = Text()
        self.write(Text.from_markup(f"\n[bold magenta]{label}[/]  "))

    def append_stream(self, fragment: str) -> None:
        if not fragment:
            return
        self._stream_buffer.append(fragment)
        # RichLog does not support in-place rewrite of the last line portably;
        # write fragments continuously for smooth streaming.
        self.write(Text(fragment))

    def end_assistant(self) -> None:
        self._streaming = False


class HotSwapApp(App[None]):
    """Async event-driven TUI coordinating SessionEngine + NetworkPipeline."""

    TITLE = "Hot-Swap REPL"
    SUB_TITLE = "zero-token-burn cross-model session harness"
    CSS = """
    Screen {
        layout: horizontal;
    }
    #sidebar {
        width: 34;
        min-width: 28;
        height: 1fr;
        border: tall $accent;
        padding: 1 1;
        background: $surface;
    }
    #main {
        width: 1fr;
        height: 1fr;
    }
    #feed {
        height: 1fr;
        border: tall $primary;
        padding: 0 1;
    }
    #input {
        dock: bottom;
        height: 3;
        margin: 0 0;
    }
    .swap-blink #sidebar {
        border: tall yellow;
        background: #1a3d1a;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+s", "swap_cycle", "Cycle swap", show=True),
        Binding("f1", "help", "Help", show=True),
    ]

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.engine: Any = None
        self.pipeline: Optional[NetworkPipeline] = None
        self._stream_task: Optional[asyncio.Task[None]] = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield StatusSidebar(id="sidebar")
            with Vertical(id="main"):
                yield ChatFeed(id="feed")
                yield Input(
                    placeholder="Message, or /swap gpt|claude|grok|open …",
                    id="input",
                )
        yield Footer()

    async def on_mount(self) -> None:
        try:
            from hotswap_repl._engine import SessionEngine
        except ImportError as exc:
            self.notify(
                "Rust extension not built. Run: maturin develop",
                severity="error",
                timeout=10,
            )
            feed = self.query_one("#feed", ChatFeed)
            feed.add_system(
                f"Failed to import SessionEngine: {exc}\n"
                "Build with `maturin develop` then relaunch.",
                style="bold red",
            )
            return

        self.engine = SessionEngine(self.settings.system_prompt)
        if self.settings.anthropic_tokenizer_path:
            try:
                self.engine.load_hf_tokenizer(
                    self.settings.anthropic_tokenizer_path, "anthropic"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("anthropic tokenizer load failed: %s", exc)
        if self.settings.openweight_tokenizer_path:
            try:
                self.engine.load_hf_tokenizer(
                    self.settings.openweight_tokenizer_path, "openweight"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("openweight tokenizer load failed: %s", exc)

        self.pipeline = NetworkPipeline(engine=self.engine, settings=self.settings)
        await self.pipeline.open()
        self.pipeline.set_active(self.settings.default_provider)

        feed = self.query_one("#feed", ChatFeed)
        feed.add_system(
            "Hot-Swap REPL ready. Dual-write prefix caching armed.\n" + HELP_TEXT,
            style="dim",
        )
        self._refresh_sidebar()
        self.query_one("#input", Input).focus()

        # Speculative warmup of system prompt across providers.
        self.run_worker(self._initial_warmup(), exclusive=False, name="warmup")

    async def _initial_warmup(self) -> None:
        if self.pipeline is None or self.engine is None:
            return
        # Seed system into master so both caches see a stable prefix.
        if self.settings.system_prompt:
            self.engine.append_turn("system", self.settings.system_prompt)
        results = await self.pipeline.dual_write_warmup()
        feed = self.query_one("#feed", ChatFeed)
        ok = sum(1 for r in results if r.ok)
        feed.add_system(
            f"Dual-write warmup complete: {ok}/{len(results)} providers warm.",
            style="green" if ok else "yellow",
        )
        self._refresh_sidebar()

    def _refresh_sidebar(self) -> None:
        if self.engine is None or self.pipeline is None:
            return
        try:
            sidebar = self.query_one("#sidebar", StatusSidebar)
        except NoMatches:
            return
        sidebar.update_from_engine(self.engine, self.pipeline.active)

    @on(Input.Submitted, "#input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = (event.value or "").strip()
        event.input.value = ""
        if not raw:
            return
        if raw.startswith("/"):
            await self._handle_command(raw)
            return
        await self._handle_chat(raw)

    async def _handle_command(self, raw: str) -> None:
        feed = self.query_one("#feed", ChatFeed)
        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"/quit", "/exit", "/q"}:
            await self.action_quit()
            return

        if cmd in {"/help", "/?"}:
            feed.add_system(HELP_TEXT)
            return

        if cmd == "/clear":
            if self.engine is not None:
                self.engine.clear()
                if self.settings.system_prompt:
                    self.engine.append_turn("system", self.settings.system_prompt)
            feed.clear()
            feed.add_system("Session cleared. Prefix caches reset locally.")
            self._refresh_sidebar()
            return

        if cmd == "/stats":
            if self.engine is None:
                return
            stats = self.engine.stats()
            lines = "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items()))
            feed.add_system(f"Engine stats:\n{lines}")
            return

        if cmd == "/warmup":
            if self.pipeline is None:
                return
            feed.add_system("Running dual-write warmup…", style="yellow")
            results = await self.pipeline.dual_write_warmup()
            for r in results:
                status = "ok" if r.ok else f"fail ({r.error})"
                feed.add_system(
                    f"  {r.provider.value}: {status}  ~{r.tokens_hint} tok  "
                    f"{r.latency_ms:.0f}ms"
                )
            self._refresh_sidebar()
            return

        if cmd == "/swap":
            if not args:
                feed.add_system("Usage: /swap gpt|claude|grok|open", style="red")
                return
            await self._do_swap(args[0])
            return

        feed.add_system(f"Unknown command: {cmd}. Try /help", style="red")

    async def _do_swap(self, target: str) -> None:
        if self.pipeline is None or self.engine is None:
            return
        feed = self.query_one("#feed", ChatFeed)
        try:
            pid = self.pipeline.set_active(target)
        except ValueError as exc:
            feed.add_system(str(exc), style="red")
            return

        # Blink UI to verify hot-swap.
        sidebar = self.query_one("#sidebar", StatusSidebar)
        self.add_class("swap-blink")
        sidebar.swap_flash = True
        self._refresh_sidebar()
        feed.add_system(
            f"⚡ Hot-swapped → {pid.display_name}  "
            f"(family={pid.engine_family}, streaming cursor retargeted)",
            style="bold yellow",
        )
        self.post_message(SwapBlink(pid))
        await asyncio.sleep(0.45)
        sidebar.swap_flash = False
        self.remove_class("swap-blink")
        # Speculatively re-warm the new dormant set.
        self.run_worker(self.pipeline.dual_write_warmup(), exclusive=False, name="post-swap-warmup")

    async def _handle_chat(self, user_text: str) -> None:
        if self.pipeline is None or self.engine is None:
            return
        if self._busy:
            self.notify("Already streaming — wait for the current reply.", severity="warning")
            return

        feed = self.query_one("#feed", ChatFeed)
        feed.add_user(user_text)
        label = self.pipeline.active.display_name
        feed.begin_assistant(label)
        self._busy = True
        self._refresh_sidebar()

        try:
            async for chunk in self.pipeline.chat_turn(user_text, warmup=True):
                if chunk.text:
                    feed.append_stream(chunk.text)
                if chunk.done:
                    break
        except Exception as exc:  # noqa: BLE001
            feed.add_system(f"\nStream error: {exc}", style="bold red")
            logger.exception("chat stream failed")
        finally:
            feed.end_assistant()
            self._busy = False
            self._refresh_sidebar()

    def action_help(self) -> None:
        try:
            feed = self.query_one("#feed", ChatFeed)
            feed.add_system(HELP_TEXT)
        except NoMatches:
            pass

    def action_swap_cycle(self) -> None:
        if self.pipeline is None:
            return
        order = [
            ProviderId.ANTHROPIC,
            ProviderId.OPENAI,
            ProviderId.GROK,
            ProviderId.OPENWEIGHT,
        ]
        try:
            idx = order.index(self.pipeline.active)
        except ValueError:
            idx = 0
        nxt = order[(idx + 1) % len(order)]
        self.run_worker(self._do_swap(nxt.value), exclusive=True, name="swap-cycle")

    async def action_quit(self) -> None:
        if self.pipeline is not None:
            await self.pipeline.close()
        self.exit()

    async def on_unmount(self) -> None:
        if self.pipeline is not None:
            await self.pipeline.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    settings = get_settings()
    app = HotSwapApp(settings=settings)
    app.run()


if __name__ == "__main__":
    main()
