#!/usr/bin/env bash
# ── run.sh — Start the SkinScan AI server ──────────────────────────────────
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
# This script:
#   1. Creates a Python virtual environment (if not already present)
#   2. Installs backend dependencies
#   3. Starts the FastAPI server on http://localhost:8000
# ──────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/backend"

# ── Python venv ────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment…"
  python3 -m venv .venv
fi

source .venv/bin/activate

# ── Install deps ───────────────────────────────────────────────────────────
echo "📦 Installing dependencies…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ── Model check ────────────────────────────────────────────────────────────
if [ ! -f "models/convnext_best.pth" ]; then
  echo ""
  echo "⚠️  WARNING: No model checkpoint found!"
  echo "   Place your trained 'convnext_best.pth' in:  backend/models/"
  echo "   Train the model using the Colab notebook first."
  echo "   The server will start but /predict will return 503."
  echo ""
fi

mkdir -p models

# ── Start server ───────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting SkinScan AI server…"
echo "   → http://localhost:8000        (Frontend UI)"
echo "   → http://localhost:8000/docs   (API docs)"
echo "   → http://localhost:8000/health (Health check)"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
