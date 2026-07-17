#!/usr/bin/env bash
# Ensure SemVer version strings stay in lockstep across package manifests.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cargo_ver="$(sed -nE 's/^version = "([^"]+)"/\1/p' Cargo.toml | head -n1)"
pyproject_ver="$(sed -nE 's/^version = "([^"]+)"/\1/p' pyproject.toml | head -n1)"
python_ver="$(sed -nE 's/^__version__ = "([^"]+)".*/\1/p' python/ucp/__init__.py | head -n1)"
manifest_ver="$(sed -nE 's/^[[:space:]]*".": "([^"]+)"/\1/p' .release-please-manifest.json | head -n1)"

semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)([+-][0-9A-Za-z.-]+)?$'

echo "Cargo.toml:                 $cargo_ver"
echo "pyproject.toml:             $pyproject_ver"
echo "python/ucp/__init__.py:     $python_ver"
echo ".release-please-manifest:   $manifest_ver"

for v in "$cargo_ver" "$pyproject_ver" "$python_ver" "$manifest_ver"; do
  if [[ ! "$v" =~ $semver_re ]]; then
    echo "::error::Not a valid SemVer version: $v"
    exit 1
  fi
done

if [[ "$cargo_ver" != "$pyproject_ver" || "$cargo_ver" != "$python_ver" || "$cargo_ver" != "$manifest_ver" ]]; then
  echo "::error::SemVer mismatch across package manifests."
  echo "All of Cargo.toml, pyproject.toml, python/ucp/__init__.py, and"
  echo ".release-please-manifest.json must share the same version."
  exit 1
fi

echo "OK: all manifests at v$cargo_ver"
