#!/usr/bin/env bash
# scripts/run_features.sh
# PS 26152 — Criminal Network Analysis System
# -------------------------------------------
# Runs the four new analytical feature modules offline.
# Writes all outputs to demo_cache/feature_outputs/.
#
# Usage:
#   bash scripts/run_features.sh
#
# Env vars (optional):
#   SKIP_VENV=1   — skip venv, use system Python
#   FEAT_SEED=42  — override random seed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${FEAT_SEED:-42}"

echo "==========================================="
echo "  CNA — Feature Pipeline"
echo "  Repo: $REPO_ROOT"
echo "  Seed: $SEED"
echo "==========================================="

# ── Python executable ─────────────────────────────────────────────────────────
VENV_DIR="$REPO_ROOT/.bench_venv"
if [ "${SKIP_VENV:-0}" = "1" ]; then
    PYTHON="python3"
else
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    PYTHON="$VENV_DIR/bin/python"
    if ! "$PYTHON" -c "import numpy" 2>/dev/null; then
        "$PYTHON" -m pip install --quiet "numpy>=1.24"
    fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# ── Step 1: Ensure demo data exists ──────────────────────────────────────────
echo ""
echo "[1/6] Ensuring demo data ..."
"$PYTHON" scripts/generate_demo_data.py

# ── Step 2: Ensure demo cache ────────────────────────────────────────────────
echo ""
echo "[2/6] Ensuring demo cache ..."
"$PYTHON" scripts/prepare_demo_cache.py

# ── Step 3: Event-Causality Chains ───────────────────────────────────────────
echo ""
echo "[3/6] Building event-causality chains ..."
"$PYTHON" -m M3_feature.event_causality build_all

# ── Step 4: Ghost-Node Finder ────────────────────────────────────────────────
echo ""
echo "[4/6] Running ghost-node finder ..."
"$PYTHON" -m M3_feature.ghost_node suggest_all

# ── Step 5: Temporal Motif Flagger ───────────────────────────────────────────
echo ""
echo "[5/6] Flagging temporal motifs ..."
"$PYTHON" -m M3_feature.motif_flagger flag_all

# ── Step 6: Auto-Brief ───────────────────────────────────────────────────────
echo ""
echo "[6/6] Generating tactical brief ..."
"$PYTHON" -m M6_feature.auto_brief export case_A

echo ""
echo "==========================================="
echo "  ✓ All features complete."
echo "  Outputs in: $REPO_ROOT/demo_cache/feature_outputs/"
ls "$REPO_ROOT/demo_cache/feature_outputs/" 2>/dev/null || true
echo "==========================================="
