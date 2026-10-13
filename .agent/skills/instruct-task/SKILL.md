---
name: instruct-task
description: Konfigurasi, tokenisasi, dan pelatihan model InstructText (SFT). Gunakan saat tipe task adalah InstructTextTask atau ChatTask, atau saat melakukan supervised fine-tuning pada data instruction-following.
---

# Pelatihan Task Instruct

## Kapan digunakan

Gunakan skill ini saat melatih model pada data instruction-following (SFT). Ini mencakup tipe `InstructTextTask` dan `ChatTask` dalam pipeline.

## Langkah-langkah pipeline

1. `text_trainer.py --task-type InstructTextTask` mengatur semuanya
2. `instruct_config.py` memilih konfigurasi (LR, batch, jumlah GPU) berdasarkan jumlah parameter model
3. `tokenize_instruct.py` memisahkan data menjadi train/dev, tokenisasi via Axolotl
4. `train_instruct.py` menjalankan pelatihan dengan HuggingFace `Trainer`

## Tingkatan konfigurasi utama

Konfigurasi dipilih secara otomatis berdasarkan ukuran model. Lihat [instruct_config.py](scripts/instruct_config.py) untuk dict `INSTRUCT_CONFIG` lengkap.

| Ukuran | LR | GPU | LoRA | Terdistribusi |
|--------|-----|-----|------|---------------|
| <1B | 1e-4 | 1 | Tidak | DDP |
| 1-9B | 7.5e-5 ~ 3.5e-5 | 1-2 | Tidak | DDP |
| 9-15B | 1e-4 | 2-4 | Ya | DDP/DS |
| 15-80B | 8e-5 | 4-8 | Ya | DeepSpeed |

## Format dataset

Dua format yang didukung (diatur via `dataset_type`):

**Custom instruct:** `{"instruction": "...", "input": "...", "output": "..."}`

**Chat template:** `{"messages": [{"role": "user", "content": "..."}, ...]}`

## Hal yang perlu diperhatikan

- LR Finder hanya berjalan saat `checking_mode="first_time"` dan dilewati untuk DeepSpeed
- Packing dinonaktifkan saat `disable_fa=True` atau untuk model OPT
- `text_trainer.py` menjalankan eksplorasi LR multi-fase untuk Instruct (berbeda dengan GRPO yang single-run)
- `reg_ratio` dari `text_trainer.py` mengalikan learning rate final

## Cara memodifikasi

- **Learning rate**: Edit `INSTRUCT_CONFIG` di `instruct_config.py` atau tambahkan LR spesifik model ke `lrs_lookup.py`
- **Batch size**: Edit `INSTRUCT_CONFIG`. Override khusus di `FIXED_BS_CONFIG` untuk model tertentu
- **Tambah model**: Tambahkan ke `models.json`, opsional tambahkan LR spesifik model ke `lrs_lookup.py`
