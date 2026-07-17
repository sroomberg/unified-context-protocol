# Versioning

This project uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Given a version `MAJOR.MINOR.PATCH`:

| Change | Bump | How to signal |
|---|---|---|
| Breaking API change | **MAJOR** | Conventional commit footer `BREAKING CHANGE:` or `feat!:` / `fix!:` |
| Backward-compatible feature | **MINOR** | PR/commit title `feat:` |
| Backward-compatible bug fix | **PATCH** | PR/commit title `fix:` |

While the crate is still `0.x` (pre-1.0):

- `feat:` bumps **MINOR** (`0.1.0` → `0.2.0`)
- `fix:` bumps **PATCH** (`0.1.0` → `0.1.1`)
- breaking changes also bump **MINOR** (not MAJOR) until `1.0.0`

## Version sources (must match)

| File | Field |
|---|---|
| `Cargo.toml` | `package.version` (source of truth for Rust / `ucp::VERSION`) |
| `pyproject.toml` | `project.version` |
| `python/ucp/__init__.py` | `__version__` |
| `.release-please-manifest.json` | `"."` |

`scripts/check_versions.sh` (and the **Version sync** GitHub Action) fail the build if these drift.

## Automation

[release-please](https://github.com/googleapis/release-please) reads Conventional Commits on `master`, opens a Release PR that bumps every version source + `CHANGELOG.md`, and publishes a GitHub Release / `vX.Y.Z` tag when that PR merges.
