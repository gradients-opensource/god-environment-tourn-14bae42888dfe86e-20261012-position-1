---
name: autotune
description: Alur Kerja (Workflow) Otonom AI untuk Men-Tuning Hyperparameter LLM
---

## Skenario Utilitas
Gunakan *skill* ini saat Pengguna memerintahkan Anda untuk mencari konfigurasi yang paling maksimal bagi satu tipe model spesifik menggunakan **Instruct, DPO,** atau **GRPO**.

## Prasyarat Lingkungan Kerja
Tergantung pada `AutoTuner_MasterText` MCP Server.

## Langkah-langkah Eksekusi

1. **Pemetaan Metadata**
   - Cari tahu kelompok parameter model (contoh: *1-2 Billion*).
   - Jalankan `read_config` ke skrip spesifik target.

2. **Eksekusi Dry Run**
   - Gunakan `run_training_trial` dengan `--max_steps 100`.

3. **Penawar OOM (Out-Of-Memory)**
   - Jika `OOM`: matikan `--packing` (Instruct), turunkan `vllm_gpu_memory_utilization` (GRPO), atau belah dua `batch_size`. Lakukan ulang Dry Run.

4. **Kalibrasi Learning Rate**
   - Tarik data uji menggunakan `check_wandb_run` atau `read_latest_eval_loss`.
   - Gunakan `modify_config_regex` jika grafik meledak (`NaN` -> turun 10x) atau stagnan (-> naik 2x).

5. **Dokumentasi & Deploy**
   - Laporkan hasil akhir via Markdown.
   - Jangan jalankan eksekusi akhir manual tanpa persetujuan. Jika lulus, unggah dengan `upload_to_huggingface`.
