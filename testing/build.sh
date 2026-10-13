#!/bin/bash
# Build semua Docker images yang dibutuhkan (downloader, trainer, uploader).
# Stop di build pertama yang gagal supaya ga lanjut dengan image setengah jadi.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo " Building Docker images untuk text-lambada..."
echo "================================================"

cd "$REPO_ROOT"

echo "[1/3] Building trainer-downloader..."
docker build -f dockerfiles/trainer-downloader.dockerfile -t trainer-downloader .

echo "[2/3] Building standalone-text-trainer..."
docker build -f dockerfiles/standalone-text-trainer.dockerfile -t standalone-text-trainer .

echo "[3/3] Building hf-uploader..."
docker build -f dockerfiles/hf-uploader.dockerfile -t hf-uploader .

echo ""
echo "✅ Build selesai! Ketiga image siap:"
echo "   - trainer-downloader"
echo "   - standalone-text-trainer"
echo "   - hf-uploader"
echo ""
echo "   Cek: docker images | grep -E 'trainer-downloader|standalone-text-trainer|hf-uploader'"
