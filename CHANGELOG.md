# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Rustdoc / Python docstrings on major public APIs
- Keep a Changelog policy and release-please automation
- Explicit SemVer policy (`VERSIONING.md`), `ucp::VERSION`, and version-sync CI

## [0.1.0] - 2026-07-12

### Added

- Pure Rust `SessionEngine` with append-only master context and dual tokenizer alignment
- Provider payload builders for OpenAI, Anthropic, Grok, and open-weight templates
- Optional PyO3 / Python bindings behind the `python` Cargo feature
- CI workflow for Rust and Python matrix tests

[Unreleased]: https://github.com/sroomberg/unified-context-protocol/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sroomberg/unified-context-protocol/releases/tag/v0.1.0
