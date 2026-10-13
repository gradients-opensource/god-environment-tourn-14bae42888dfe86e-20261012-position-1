---
name: grpo-task
description: Konfigurasi, tokenisasi, dan pelatihan model GRPO (Group Relative Policy Optimization) dengan fungsi reward kustom. Gunakan saat tipe task adalah GrpoTask, atau saat melatih dengan optimasi berbasis reward.
---

# Pelatihan Task GRPO

## Kapan digunakan

Gunakan skill ini saat melatih dengan fungsi reward (GRPO). Pipeline menggunakan `GRPOTrainer` dari TRL dengan vLLM untuk generasi.

## Langkah-langkah pipeline

1. `text_trainer.py --task-type GrpoTask` mengatur semuanya
2. `grpo_config.py` memilih konfigurasi berdasarkan jumlah parameter model
3. `tokenize_grpo.py` memisahkan data, mengadaptasi kolom prompt
4. `train_grpo.py` memvalidasi fungsi reward, menyiapkan GRPOTrainer, melatih

## Tingkatan konfigurasi utama

Lihat [grpo_config.py](scripts/grpo_config.py) untuk dict `GRPO_CONFIG` lengkap.

| Ukuran | LR | GPU | LoRA | vLLM |
|--------|-----|-----|------|------|
| <2B | 8e-6 | 1 | Tidak | Ya (0.4) |
| 2-9B | 8e-6 ~ 6e-6 | 2-4 | Ya | Ya |
| 9-15B | 6e-6 ~ 5e-6 | 4 | Ya | Ya/Tidak |
| 15-80B | 5e-6 ~ 3e-6 | 4-8 | Ya | Tidak (4-bit) |

## Format dataset

Membutuhkan field `prompt` (bisa dikonfigurasi via `dataset_type.field_prompt`). Kolom `extra_data` bersifat opsional.

```json
{"prompt": "Tulis puisi tentang laut", "extra_data": {"max_words": 50}}
```

## Fungsi reward

Didefinisikan sebagai string kode Python di `dataset_type.reward_functions`. Setiap fungsi harus:
- Menerima `completions` (list string), opsional `extra_data`
- Mengembalikan list angka (panjang sama dengan completions)
- Divalidasi sebelum pelatihan via `validate_reward_function()`

## Hal yang perlu diperhatikan

- GRPO melewati eksplorasi LR multi-run — `text_trainer.py` langsung mengeset `mode="finish"`
- vLLM berjalan dalam mode **colocate** dan otomatis dinonaktifkan saat OOM
- Model ≥15B sering tidak bisa menggunakan vLLM; mereka fallback ke HF generate
- LR Finder berbasis reward (`lr_finder_grpo.py`), bukan Leslie Smith — dan hanya berjalan saat `find_lk_lr=True` dan `gpu_count=1`
- Prompt dipotong dari kiri ke `max_prompt_length` (default 512)
- `num_generations=2` secara default — mempengaruhi perhitungan steps-per-epoch
- Fungsi reward yang lambat (langcheck, detoxify, textstat) memicu pengurangan batch size

## Cara memodifikasi

- **Learning rate**: Edit `GRPO_CONFIG` di `grpo_config.py` atau `lrs_lookup.py` (`get_grpo_lr()`)
- **Memori vLLM**: Edit `vllm_gpu_memory_utilization` di konfigurasi tingkatan
- **Generasi**: Ubah `num_generations` di `grpo_config.py` run_config (default 2)
- **Nonaktifkan vLLM**: Setel `"use_vllm": False` di tingkatan atau biarkan otomatis dinonaktifkan saat OOM
