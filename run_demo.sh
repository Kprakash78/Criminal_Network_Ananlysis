#!/bin/bash
# ===========================================================================
# run_demo.sh — Criminal Network Analysis System Demo Launcher
# PS 26152 — AI-Powered Criminal Network Analysis System
#
# Sets up environment, generates demo outputs, and launches the
# investigation dashboard.
#
# Usage:
#   chmod +x run_demo.sh && ./run_demo.sh
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Criminal Network Analysis — Demo Setup"
echo "========================================"
echo ""

# --- Step 1: Create/activate virtual environment ---
echo "[1/5] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi
source .venv/bin/activate
echo "  Activated: $(which python3)"

# --- Step 2: Install dependencies ---
echo ""
echo "[2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet scikit-learn 2>/dev/null || echo "  (scikit-learn optional, TF-IDF fallback)"
pip install --quiet cryptography 2>/dev/null || echo "  (cryptography optional, HMAC fallback)"
echo "  Dependencies installed"

# --- Step 3: Generate demo dataset outputs ---
echo ""
echo "[3/5] Generating demo dataset outputs..."
python3 demo_dataset/generate_demo_outputs.py
echo "  Demo M1 outputs generated"

# --- Step 4: Run Top-3 suspect analysis ---
echo ""
echo "[4/5] Running Top-3 suspect analysis..."
python3 -m M3_feature.top3
echo "  Top-3 results generated"

# --- Step 5: Launch Streamlit dashboard ---
echo ""
echo "[5/5] Launching Investigation Dashboard..."
echo "========================================"
echo "  Dashboard URL: http://localhost:8501"
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

CRIMINAL_USE_REAL_MODULES=0 streamlit run M6_feature/search_ui.py --server.headless true
