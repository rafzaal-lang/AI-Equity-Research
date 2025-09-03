#!/usr/bin/env bash
# Setup script for AI Equity Research Platform

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "🔧 Setting up AI Equity Research Platform..."

# --------------------------------------------------------------------
# Python detection
# --------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "❌ Python not found. Install Python 3.11+."
  exit 1
fi

PY_VER="$($PY - <<'PYCODE'
import sys
print(".".join(map(str, sys.version_info[:3])))
PYCODE
)"

REQ_MAJOR=3
REQ_MINOR=11
MAJOR="${PY_VER%%.*}"
MINOR="$(python - <<'PYCODE'
import sys
v=sys.version_info
print(v.minor)
PYCODE
)"

if (( MAJOR < REQ_MAJOR )) || { (( MAJOR == REQ_MAJOR )) && (( MINOR < REQ_MINOR )); }; then
  echo "❌ Python 3.11+ required. Found: $PY_VER"
  exit 1
fi
echo "✅ Python version check passed: $PY_VER"

# --------------------------------------------------------------------
# Virtual environment
# --------------------------------------------------------------------
if [[ ! -d "venv" ]]; then
  echo "📦 Creating virtual environment..."
  $PY -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
echo "✅ Virtual environment activated (venv)"

# --------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------
python -m pip install --upgrade pip
echo "✅ Pip upgraded"

echo "📚 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# --------------------------------------------------------------------
# Project directories
# --------------------------------------------------------------------
mkdir -p templates static reports logs data
echo "✅ Created required directories"

# --------------------------------------------------------------------
# Env file
# --------------------------------------------------------------------
if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
    echo "⚠️  Please edit .env and add your API keys"
  elif [[ -f ".env.template" ]]; then
    cp .env.template .env
    echo "✅ Created .env from .env.template"
    echo "⚠️  Please edit .env and add your API keys"
  else
    echo "⚠️  Please create a .env file with your API keys (see .env.example)"
  fi
fi

# --------------------------------------------------------------------
# NLTK/TextBlob data (best-effort)
# --------------------------------------------------------------------
python - <<'PYCODE'
try:
    import nltk
    for pkg in ["punkt","vader_lexicon","wordnet","omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    print("✅ NLTK data downloaded (best-effort)")
except Exception:
    print("ℹ️  NLTK not available; skipping corpus download")

try:
    import textblob
    from textblob import download_corpora
    download_corpora.download_all()
    print("✅ TextBlob corpora downloaded (best-effort)")
except Exception:
    print("ℹ️  TextBlob corpora not downloaded; continuing")
PYCODE

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1) Edit .env and add your API keys"
echo "2) Put your UI HTML in templates/index.html (or keep the default)"
echo "3) Start the server:  source venv/bin/activate && python main.py"
echo ""
echo "🚀 Ready to launch!"
