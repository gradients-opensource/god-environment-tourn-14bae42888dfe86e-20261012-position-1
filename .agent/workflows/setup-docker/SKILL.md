---
name: setup-docker
description: Alur Kerja (Workflow) Otonom AI untuk Pemasangan Infrastruktur Docker GPU
---

## Skenario
*Skill* khusus menangani komplikasi pembangunan lapisan sterilisasi (seperti *CUDA Toolkit clash*, *Wheel build* yang putar-hilang, dll) demi menjamin kelancaran perintah eksekusi terisolasi, atau bila dicetus oleh `/setup-docker`.

## Langkah-langkah Eksekusi

1. **Tinjauan Taktis Keras (Hardware Core Probing)**
   - Tembakkan kompresi radar lokal: Lakukan eksekusi `nvidia-smi` dan `nvcc --version`. Kenali selisih ketersediaan dan versi bawaan OS Host aslinya.
   
2. **Harmonisasi Skrip Pelapis Kontainer (Container Version Sync)**
   - Pindai sasaran struktur file kompilasi `dockerfiles/standalone-text-trainer.dockerfile` (segenap modul rujukan lain).
   - Timpa rujukan fundamental *image build* `FROM nvidia/cuda:XX...` di baris prolog apabila rentang versinya berlawanan melampaui batasan keras fisik/GPU Pengguna yang diamati di eksekusi awal Langkah 1.

3. **Inisialisasi Reaktor Kompilasi Berbatas Waktu (Watchdog Build Sequence)**
   - Jalankan peleburan konstruksi wujud penengah aslinya: `docker build -f dockerfiles/standalone-text-trainer.dockerfile -t trainer-env .`
   - Amati proses rotasi roda memori kompilasi rumit miliki pustaka *Ninja FlashAttention* / *DeepSpeed OPs*. Jika waktu rotasinya membeku tak alami menginjak limit 30 menit, jangan buang aset mesin; potong paksa rutenya, manipulasi paksa kode *Dockerfile* terkait guna melipir jalur unduh versi matang (`pre-compiled Wheel release binary`) ketimbang menggiling dari kode asalnya (Github).

4. **Integrasi Gerbang Jembatan Taut Virtual (Mount Bridge Deployment)**
   - Lempar operasionalisasi ke lapisan siluman penarget kontainer (`docker run -it`). Tambatkan persis parameter volume `-v` pada pangkal root ke repositori asli.
   - Sampaikan notifikasi resi peresmian sukses pada terminal Pengguna.
