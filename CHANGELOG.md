# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/sroomberg/unified-context-protocol/compare/v0.1.0...v0.1.1) (2026-07-21)


### Added

* enforce SemVer across manifests and release automation ([96b4023](https://github.com/sroomberg/unified-context-protocol/commit/96b402321b0743b17f1fc3aee0ab4839d7419a93))


### Changed

* add rustdoc and Python docstrings on public APIs ([1ebec5a](https://github.com/sroomberg/unified-context-protocol/commit/1ebec5a0792850152954b7a8fe59214ce4a72dc9))

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
