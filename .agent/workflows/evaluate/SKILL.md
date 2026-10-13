---
name: evaluate
description: Alur Kerja (Workflow) Otonom AI untuk Ujian Kelulusan Model (Evaluation)
---

## Skenario
Jalankan komando ini saat sesi *Training 100% Epoch* tamat, atau Pengguna memanggil pelatuk perintah `/evaluate` pada suatu lokasi *checkpoint / output_dir* lampau untuk memeriksa level repetisi gaya bicara demi menangkal *Overfitting*.

## Langkah-langkah Eksekusi

1. **Penggabungan LoRA (Merging)**
   - Jika sistem mendeteksi rilis arsitektur LoRA (*Adapter Weights*), siapkan dan inisiasikan injeksi skrip memori pembantu semacam `peft.merge_and_unload()` ke dalam *Base Model*.

2. **Eksekusi Perisai Evaluasi (Evaluation Run)**
   - Prioritaskan memanggil pustaka awan *lm-eval* (Harness) jika ada dalam *requirements*.
   - Bila absen, formulasikan modul pembangkit luaran teks Python (mengacu pada *greedy decoding*) berisi setidaknya 20 pertanyaan dari bingkai acak *validation set*.

3. **Inspektur Performa Bahasa (Alignment Test)**
   - Cek tingkat pusing teks (*perplexity/repetitive response*).
   - Ukur kehadiran *EOS Token / `<|im_end|>`* untuk menggagalkan peretas respons *infinite loop*.
   
4. **Ringkasan Kartu Prestasi (Model Card)**
   - Tampilkan *Markdown Table* menguraikan komparasi *V0* lawan *V1* meliputi penurunan persentase *Eval Loss / Reward Margin* dan sertakan cuplikan spesimen percakapan interaksi tuntas.
