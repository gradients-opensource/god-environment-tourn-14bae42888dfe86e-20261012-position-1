#!/bin/bash
# ============================================================
# Baseline Stats Test
# Cek Docker images, GPU, dan environment sebelum test training
# ============================================================

echo "================================================"
echo " BASELINE STATS TEST"
echo "================================================"
echo ""

# 1. Cek Docker
echo "[1/5] Docker daemon..."
if docker info > /dev/null 2>&1; then
    echo "  ✅ Docker running"
else
    echo "  ❌ Docker tidak berjalan!"
    exit 1
fi

# 2. Cek Docker images
echo "[2/5] Docker images..."
for IMG in standalone-text-trainer trainer-downloader hf-uploader; do
    if docker image inspect "$IMG" > /dev/null 2>&1; then
        echo "  ✅ $IMG"
    else
        echo "  ❌ $IMG — belum ada! Jalankan build.sh"
    fi
done

# 3. Cek GPU
echo "[3/5] GPU (nvidia-smi)..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 \
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/  🖥️  /' \
    || echo "  ❌ GPU tidak terdeteksi di Docker"

# 4. Cek tokens
echo "[4/5] Credentials..."
if [ -n "$HF_TOKEN" ]; then
    echo "  ✅ HF_TOKEN diset"
else
    echo "  ⚠️  HF_TOKEN kosong — isi di .telegram.env"
fi
if [ -n "$WANDB_TOKEN" ]; then
    echo "  ✅ WANDB_TOKEN diset"
else
    echo "  ℹ️  WANDB_TOKEN kosong (opsional)"
fi

# 5. Cek direktori
echo "[5/5] Direktori..."
mkdir -p "$(pwd)/secure_checkpoints" "$(pwd)/outputs"
chmod 700 "$(pwd)/secure_checkpoints" "$(pwd)/outputs"
echo "  ✅ secure_checkpoints/ dan outputs/ siap"

echo ""
echo "================================================"
echo " Semua ✅ = siap test training!"
echo " Jika ada ❌, perbaiki dulu."
echo "================================================"
