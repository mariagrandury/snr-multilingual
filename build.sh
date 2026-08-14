#!/usr/bin/env bash
set -euo pipefail

# Build pipeline for Netlify (and local).
# Outputs the full site to ./site:
#   /         -> mkdocs (Material) docs
#   /slides/  -> Slidev slides built from documents/slides.md

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Installing Python doc deps"
pip install --quiet -r requirements-docs.txt

echo "==> Building MkDocs"
mkdocs build --clean

echo "==> Building Slidev slides"
cd "$ROOT/documents"
if command -v pnpm >/dev/null 2>&1; then
  pnpm install --frozen-lockfile
  pnpm run build --base /slides/
else
  npm install
  npx slidev build --base /slides/
fi
cd "$ROOT"

echo "==> Merging slides into site/slides"
rm -rf "$ROOT/site/slides"
mkdir -p "$ROOT/site/slides"
cp -r "$ROOT/documents/dist/." "$ROOT/site/slides/"

echo "==> Build complete: $ROOT/site"
