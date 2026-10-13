# Aturan Umum (General Rules) Repositori Master-Text

Ini adalah repositori sentral untuk proses Pelatihan (Training) dan Penyelarasan (Fine-Tuning) *Large Language Models (LLM)* tingkat lanjut milik pengguna. Repositori ini mendukung metode **Supervised Fine-Tuning (Instruct), Direct Preference Optimization (DPO),** dan **Generative Reward Policy Optimization (GRPO)**.

Saat beroperasi di repositori ini, pastikan Anda sebagai entitas kode mengikuti arsitektur dan hukum logika di bawah ini agar perubahan/skrip yang Anda sarankan tidak asala-asalan.

---

## 1. Arsitektur Folder Wajib Diketahui
- **`scripts/`**: INTI REPOSITORI. Semua eksekusi komputasi keras, skrip pelatihan utama (`train_*.py`), definisi parameter dinamis (`*_config.py`), dan skrip *preprocessing/tokenization* bersarang di sini. Modifikasi komputasi LLM dilakukan di sini.
- **`trainer/`**: Berfungsi sebagai antarmuka API/Server Backend yang menjembatani server (via ASGI/endpoints) dengan antrean tugas asinkron (`tasks.py`).
- **`dockerfiles/`**: Berisi manifest/fondasi lingkungan kontainer (seperti `standalone-text-trainer.dockerfile`). Asumsikan semua eksekusi Python yang aslinya berjalan akan dilakukan **di dalam selubung Docker**, bukan di lingkungan murni sistem OS host.
- **`mcp-auto-tuner/`**: Ekstensi server lokal kustom berbasis Model Context Protocol (MCP) untuk memfasilitasi *auto-tuning hyperparameter* oleh AI.

---

## 2. Aturan Merombak Parameter Konfigurasi (Tuning)
JANGAN PERNAH menyarankan pengguna mengubah kode konfigurasi (seperti nilai asal *Learning Rate* atau *Batch Size*) menggunakan metode tebak-tebakan. Ikuti hukum hierarki *Master-Text* berikut saat membedah `instruct_config.py`, `dpo_config.py`, atau `grpo_config.py`:

1. **Titik Awal (Dictionary Mapping)**: Cari referensi parameter utama berdasarkan rentang jumlah parameter (*Billion parameters*) model (misal kunci kamus `0_1_b`, `4_5_b`, `40_80_b`).
2. **Kewaspadaan Terhadap Hardcode Overrides (Sangat Penting)**: Jangan sekadar mengubah tabel *dictionary* bagian atas! Selalu baca hingga baris terbawah (di dalam fungsi `get_training_json()` atau sejenisnya). Skrip ini cerdas dan sangat sering secara sengaja **memotong (*slice*)/meng-override nilai `batch_size` dari dictionary ke persentase mutlak** untuk menangani kerentanan memori pada arsitektur tertentu (contoh: model `falcon`, `phi`, `gpt-j`, `gemma-2-9b` akan selalu dipotong paksa).
3. **Penyelamatan VRAM (*Memory Optimization*)**:
   - Jika `OOM` (*Out of Memory*) pada **Instruct**: Ingat untuk selalu mencoba menonaktifkan mode pemampatan data (`--packing False`) sebelum memotong `batch_size`.
   - Jika `OOM` pada **GRPO**: Evaluasi ulang proporsi memory peladen *Student-Teacher* lewat mengatur parameter `--vllm_gpu_memory_utilization` atau kurangi tebaran paralel generasi sel serentak (`--num_generations`). VRAM GRPO dieksploitasi dengan sangat ekstrim.
   - Pahami bahwa fitur `gradient_accumulation_steps` mayoritas dihitung matematis secara terotomatisasi di ujung skrip (misal: `int(64 / total_batch_size)` agar menyentuh angka aman 64). Jangan memaksa mendeklarasikannya secara paksa dalam tabel utama jika berlawanan dengan rumus ujung ini.

---

## 3. Pengecualian Tabel Eksternal (LRS Lookup)
Bila dalam kode akhir skrip membaca variabel saklar **`find_lk_lr` bernilai `True`**, seluruh keringat modifikasi Anda terhadap *Learning Rate (LR)* di dalam file konfigurasi akan lenyap tak berguna! Anda (AI) WAJIB menyadari bahwa skrip tersebut **memprioritaskan nilai eksternal dari fungsi yang berafiliasi di dalam `lrs_lookup.py`** (contoh fungsi: `get_instruct_lr`, `get_grpo_python_lr`, atau `get_dpo_lr`).  
> **Aksi:** Jika ini terjadi, ubah/timpa nilai keluaran yang bersumber dari tabel file `lrs_lookup.py` tersebut secara langsung, bukan malah mengotak-atik file konfigurasi utamanya.

---

## 4. Evaluasi & Logging
- **Deadline Komputasi (Early Stopping)**: Evaluasi pelatihan berjalan bukan dari sistem dasar Huggingface murni, melainkan melewati `CustomEvalSaveCallback`. Skrip *training* dibangun dengan kemampuan untuk **mendeteksi *deadline* sisa waktu** komputasi sewa. Pelatihan bisa berhenti dengan apik jika waktunya akan habis.
- **Tracking Log & Metrik**: Monitoring tren grafik *Loss/Reward* terintegrasi ke `wandb` (Weights & Biases). Saat AI Anda memanggil alat penganalisa log/Server MCP, arahkan pemonitoran ke ekstraksi nilai file `trainer_state.json` lokal.

> **Tugas Akhir untuk AI:**
> *Dalam menganalisis *bug/error traceback*, agen pemrograman AI wajib melacak alur eksekusi dari hulu ke hilir (Contoh: Parameter JSON > `*config.py` > `text_trainer.py` > eksekusi akhir loop di `train_*.py`) sebelum melontarkan asumsi perbaikan kodingan ke repositori pengguna.*
