---
name: recover
description: Alur Kerja (Workflow) Otonom AI untuk Penyelamatan Sesi Training yang Rusak / Terhenti Menengah Jalan
---

## Skenario
Penyelamatan Waktu (Time-Reversal) untuk membangkitkan komputasi pelaksana *Trainer* yang seketika rubuh di pertengahan babak (contoh: Mati listrik peladen, VRAM Overload, Interupsi Fatal Pengguna).

## Langkah-langkah Eksekusi

1. **Penyisiran Makam Direktori (Checkpoint Discovery)**
   - Inspeksi memori hirarki direktori bawah (*subtree*) akar dari `trained_model` (atau parameter `output_dir`).
   - Eksekusi pengurutan nama arsip `checkpoint-[nomor]`. Pastikan membidik dan memegang jalur rute folder yang dinomori oleh matriks langkah paling ujung (terbesar).

2. **Integritas Validator Ekstrak Data**
   - Raba keadalaman folder sasarannya untuk memastika keberadaan trio artefak absolut: `adapter_model.safetensors`, file `rng_state.pth` (Atau entitas `optimizer.bin`), serta pembukuan `trainer_state.json`.
   - Lacak nilai *steps/epochs* terakhir sebelum tergilas takdir kematian log.

3. **Injeksi Parameter Kebangkitan (Resurrection Injection)**
   - Susun dan tambahkan klausa *flag argparsers* ini ke ujung komando pengeksekusi: `--resume_from_checkpoint [jalur_direktori_sasar_tadi]`.

4. **Peluncuran Defensif Anti-OOM**
   - Bila riwayat membuktikan kematian pemicunya adalah ledakan beban raksasa `CUDA OOM`, persilakan kewenangan Anda mengintervensi parameter *batch_size* dari skrip induk (tekan setengah / sesuaikan rasio). Lancarkan skrip pemanggilan aslinya berseri.
