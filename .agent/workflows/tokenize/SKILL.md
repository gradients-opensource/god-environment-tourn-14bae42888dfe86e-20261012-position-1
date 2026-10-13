---
name: tokenize
description: Alur Kerja (Workflow) Otonom AI untuk Validasi & Tokenisasi Dataset LLM
---

## Skenario
Gunakan *skill* ini saat pengguna menyerahkan berkas dataset mentah (JSON/CSV) atau memerintahkan `/tokenize`. Tujuannya memastikan struktur teks valid menggunakan standar format yang diinginkan sebelum *training*.

## Langkah-langkah Eksekusi

1. **Validasi Struktur Mentah**
   - Lacak `datasets/*.json` atau parameter `request_path`.
   - Bedah formasi struktur larik pertama JSON (misal `{"role": "user", "content": "..."}`).
   - Jika terdapat karakter ilegal/anomali yang parah, jalankan instruksi Python internal perbaikan (linter).

2. **Eksekusi Penyelarasan (Tokenization/Preprocessing)**
   - Eksekusi komando standar: `python scripts/tokenize_instruct.py [jalur_data_opsional]` (atau `_dpo.py` / `_grpo.py`).
   - Awasi log peringatan panjang token (*"Dropped samples due to length limit"*). Jika >5%, peringatkan pengguna untuk mempertimbangkan `max_length`.

3. **Verifikasi Serah Terima (Handover)**
   - Periksa ketersediaan berkas keluaran akhir bernamakan akhir `_tokenized.json` yang memuat pilar matrik fundamental `input_ids`.
   - Sediakan format laporan Distribusi Panjang Token.
