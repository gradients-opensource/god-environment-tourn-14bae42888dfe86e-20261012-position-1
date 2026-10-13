---
description: Alur Kerja (Workflow) End-to-End Pelatihan LLM Bertaraf Master-Text
---

## Skenario Master
Terapkan urutan mutlak *Pipeline* ini saat Pengguna meminta Anda: *"Tolong laksanakan skenario lengkap dari awal sampai akhir,"* atau sebuah baris perintah tunggal bertuliskan **`/full-run`**. 

## Orkestrasi Berantai: Fase *End-to-End*
// turbo-all

1. **Fase Pra-Penyelarasan Infrastruktur (Opsional tapi Krusial)**
   - Jika repositori baru saja di-*clone* atau Pengguna menderita "*CUDA Error*", wajib eksekusi terlebih dahulu pembedahan kerangka fisik via `/setup-docker` (`.agent/skills/setup-docker/SKILL.md`).
   - Selesaikan sinkronisasi *PyTorch* lokal dan rakitan *Image Docker*-nya sebelum melangkah sejengkal pun.

2. **Fase Formasi Pustaka Data (Tokenization)**
   - Ambil raw data dari tangan Pengguna, eksekusi pemanggilan komando `/tokenize` (`.agent/skills/tokenize/SKILL.md`).
   - Apabila persentase `Dropped samples limit` berada dalam spektrum hijau (< 5%), alihkan wewenang ke tahap peluncuran eksperimental. Jangan beranjak sebelum berkas berlabel `input_ids` lahir ke dunia direktori.

3. **Fase Penjelajahan Konfigurasi Terdalam (*Golden Param* Hunt)**
   - Jalankan `/autotune` (`.agent/skills/autotune/SKILL.md`) pada serangkaian rute target (*Instruct / DPO / GRPO*).
   - Biarkan alat MCP memicu iterasi kilat *100-steps dry run*. Perangi gelombang OOM dengan menurunkan `batch_size` atau manipulasi vLLM. Perbaiki *Learning Rate* hingga alat AWAN (W&B) `check_wandb_run` memberikan lampu persetujuan konvergensi (grafik mengecil landai).
   - **TUNDA PROGRES:** Berhenti dan tunjukkan tabel angka Hyperparameter Emas Anda. Tunggu kalimat sakti "*Lanjutkan Full Run*" dari mulut Pengguna.

4. **Fase Perang Komputasi Penuh (Full-Scale Execution)**
   - Tembakkan komando eksekusi aslinya dan tunggu mesin pengkaji menyentuh kematangan `100% Epoch`.
   - **Penawar Fatal:** Apabila di tengah asimilasi komputasi mendadak peladen membatu (Sistem OOM fatal / mati mendadak), **DILARANG PANIK!** Lakukan pernapasan buatan dengan memanggil komando `/recover` (`.agent/skills/recover/SKILL.md`). Injeksi argumetasi sakti `resume_from_checkpoint` agar tidak satu detik pun waktu sewa mesin dibuang percuma. Lanjutkan siklus pelatihannya sampai finis.

5. **Fase Ujian Integritas Logika (Alignment Benchmark)**
   - Ketika model dilabeli *Done*, panggil inspektur `/evaluate` (`.agent/skills/evaluate/SKILL.md`). 
   - Gabungkan sementara struktur persendian LoRA-nya, dan paksa ia mengerjakan teks di luar wawasan datasetnya untuk membuktikan model itu bukan "beo" (menghafal mati /*overfitting*).

6. **Fase Deploy Puncak (*The Handover*)**
   - Bila segalanya sempurna dan rapor kompetensinya tidak meragukan, sulap hasil karyanya dengan fasilitas MCP internal `upload_to_huggingface`. Kirimkan seluruh direktori hasil tempaannya (*Safetensors*) tegak-lurus ke awan *Cloud HuggingFace Hub*, dan persembahkan piala pranala URLs karya ini kepada Sang Pengguna.
