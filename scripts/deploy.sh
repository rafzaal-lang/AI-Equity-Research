#!/usr/bin/env bash
# Deployment script for AI Equity Research Platform

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "🚀 Starting deployment process from: $ROOT_DIR"

# --------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------
required_files=("main.py" "requirements.txt" "Dockerfile" "render.yaml")
for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "❌ Required file missing: $file"
    exit 1
  fi
done
echo "✅ All required files found"

# Warn if OPENAI_KEY not set (app will still run with echo provider)
if [[ -z "${OPENAI_KEY:-}" ]]; then
  echo "⚠️  Warning: OPENAI_KEY not set (the app will use a local echo provider)"
fi

# Ensure basic dirs exist
mkdir -p templates static reports logs data
echo "✅ Ensured required directories"

# --------------------------------------------------------------------
# Build (optional)
# --------------------------------------------------------------------
if [[ "${1:-}" == "docker" ]]; then
  if command -v docker >/dev/null 2>&1; then
    echo "🐳 Building Docker image..."
    docker build -t ai-equity-research:latest
