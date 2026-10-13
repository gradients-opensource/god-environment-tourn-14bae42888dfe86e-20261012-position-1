---
name: dpo-task
description: Konfigurasi, tokenisasi, dan pelatihan model DPO (Direct Preference Optimization). Gunakan saat tipe task adalah DpoTask, atau saat fine-tuning dengan pasangan preferensi (chosen vs rejected).
---

# Pelatihan Task DPO

## Kapan digunakan

Gunakan skill ini saat melatih dengan data preferensi (pasangan chosen/rejected). Pipeline menggunakan `DPOTrainer` dari TRL.

## Langkah-langkah pipeline

1. `text_trainer.py --task-type DpoTask` mengatur semuanya
2. `dpo_config.py` memilih konfigurasi berdasarkan jumlah parameter model
3. `tokenize_dpo.py` memisahkan data, menghapus item kosong, mengadaptasi kolom ke format TRL
4. `train_dpo.py` menjalankan pelatihan dengan TRL `DPOTrainer`

## Tingkatan konfigurasi utama

Lihat [dpo_config.py](scripts/dpo_config.py) untuk dict `DPO_CONFIG` lengkap. LR DPO sekitar 10x lebih rendah dari Instruct.

| Ukuran | LR | GPU | LoRA | Terdistribusi |
|--------|-----|-----|------|---------------|
| <1B | 1.35e-5 | 1 | Tidak | DDP |
| 1-4B | 8.7e-6 ~ 6.5e-6 | 1-2 | Tidak/Ya | DDP |
| 4-12B | 7.5e-6 ~ 5e-6 | 4 | Ya | DDP/DS |
| 12-80B | 8.5e-6 ~ 8e-6 | 4-8 | Ya | DeepSpeed |

## Format dataset

Tiga field yang dibutuhkan (nama kolom bisa dikonfigurasi via `dataset_type`):

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

`dataset_type` memetakan: `field_prompt`, `field_chosen`, `field_rejected`.

## Hal yang perlu diperhatikan

- `tokenize_dpo.py` TIDAK melakukan tokenisasi — hanya memisahkan dan mengadaptasi kolom. DPOTrainer menangani tokenisasi secara internal
- LR Finder menggunakan **proxy SFT** (melatih pada kolom `chosen` saja) karena loss DPO penuh memerlukan logprob ref_model yang mahal
- Beberapa model perlu token bermasalah dihapus (lihat dict `REMOVE_ADD_TOKEN` di `tokenize_dpo.py`)
- `padding_free=True` diaktifkan otomatis saat flash attention tersedia
- Gradient checkpointing dinonaktifkan untuk beberapa tingkatan DeepSpeed

## Cara memodifikasi

- **Learning rate**: Edit `DPO_CONFIG` di `dpo_config.py` atau `lrs_lookup.py` (`get_dpo_lr()`)
- **Batch size**: Edit `DPO_CONFIG`. `gradient_accumulation_steps` secara otomatis menargetkan effective batch ~64
