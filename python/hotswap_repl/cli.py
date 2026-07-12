"""CLI entry without requiring the full TUI (useful for scripting / tests)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

from hotswap_repl.config import get_settings
from hotswap_repl.network import NetworkPipeline
from hotswap_repl.providers import ProviderId, build_request_body


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hotswap-repl",
        description="Cross-model hot-swapping LLM REPL harness",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("ui", help="Launch the Textual TUI (default)")

    dump = sub.add_parser("dump-payload", help="Print provider payload JSON for the current engine")
    dump.add_argument("provider", choices=["openai", "anthropic", "grok", "openweight"])
    dump.add_argument("--user", default="Hello from hotswap-repl")
    dump.add_argument("--system", default=None)

    warm = sub.add_parser("warmup", help="Run a one-shot dual-write warmup")
    warm.add_argument("--user", default="Warmup prefix document for prompt caching.")

    return p


async def _cmd_dump_payload(provider: str, user: str, system: Optional[str]) -> int:
    from hotswap_repl._engine import SessionEngine

    settings = get_settings()
    engine = SessionEngine(system or settings.system_prompt)
    engine.append_turn("system", system or settings.system_prompt)
    engine.append_turn("user", user)
    pid = ProviderId.parse(provider)
    body = build_request_body(pid, engine, settings, stream=False)
    print(json.dumps(body, indent=2))
    return 0


async def _cmd_warmup(user: str) -> int:
    from hotswap_repl._engine import SessionEngine

    settings = get_settings()
    engine = SessionEngine(settings.system_prompt)
    engine.append_turn("system", settings.system_prompt)
    engine.append_turn("user", user)
    async with NetworkPipeline(engine=engine, settings=settings) as pipe:
        results = await pipe.dual_write_warmup()
    for r in results:
        status = "OK" if r.ok else f"FAIL:{r.error}"
        print(f"{r.provider.value:12} {status:40} tokens≈{r.tokens_hint}  {r.latency_ms:.0f}ms")
    return 0 if any(r.ok for r in results) or not results else 1


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    # Default to UI when no args.
    if not argv:
        argv = ["ui"]
    args = parser.parse_args(argv)

    if args.command in (None, "ui"):
        from hotswap_repl.app import main as ui_main

        ui_main()
        return

    if args.command == "dump-payload":
        raise SystemExit(asyncio.run(_cmd_dump_payload(args.provider, args.user, args.system)))

    if args.command == "warmup":
        raise SystemExit(asyncio.run(_cmd_warmup(args.user)))

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
