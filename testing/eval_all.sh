#!/bin/bash
# ============================================================
# Eval All — Jalankan semua test secara berurutan
# Uncomment test yang mau dijalankan
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo " EVAL ALL — Starting all tests"
echo "================================================"
echo ""

# --- Qwen2.5-1.5B ---
echo "[1] Qwen2.5-1.5B — InstructText"
bash "$SCRIPT_DIR/413f24dc_instruct.sh"
echo ""

# echo "[2] Qwen2.5-1.5B — DPO"
# bash "$SCRIPT_DIR/b2c3d4e5_dpo.sh"
# echo ""

# echo "[3] Qwen2.5-1.5B — GRPO"
# bash "$SCRIPT_DIR/c4d5e6f7_grpo.sh"
# echo ""

# --- Tambah model lain di bawah sini ---
# echo "[4] Qwen2.5-7B — InstructText"
# bash "$SCRIPT_DIR/xxxxx_instruct.sh"

echo ""
echo "================================================"
echo " EVAL ALL — Selesai!"
echo "================================================"
