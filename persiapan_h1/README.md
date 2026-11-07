# Paket persiapan hari H (GEMASTIK XIX, Data Mining)

Tiga script, tiga langkah. Semuanya aman dijalankan ulang.

| Script | Dijalankan di | Internet | Fungsi |
|---|---|---|---|
| `01_download_all.py` | Laptop sendiri | **Perlu** | Unduh semua: wheel Python, model, leksikon, data kecil, docs → satu folder `bundle` |
| `02_install_offline.ps1` / `.bat` | PC panitia (dan latihan di laptop) | **Tidak** | Pasang Python + venv + paket dari bundle, salin model, buat `env.ps1`, verifikasi |
| `03_verify_offline.py` | Otomatis dari langkah 2 | Tidak | Cek import, GPU, presisi, dan **ukur kecepatan nyata** (teks/detik, menit per 500k baris) |

## Langkah

**1. Di laptop, ada internet** (pakai Python 3.12)

```bash
python 01_download_all.py --out D:\comp\bundle --check
python 01_download_all.py --out D:\comp\bundle --profile plus
```

- `--check` hanya memeriksa semua link dan ID model (tanpa unduh) dan mencetak ukuran total.
- Profil: `core` (wajib, sekitar 8 GB), `plus` (+ model dan data tambahan, sekitar 12 GB), `all` (+ LLM lokal, fastText, dll, sekitar 30 GB).
- `--only F4,D1` mengunduh item tertentu. Kode item sama dengan tabel di laporan.
- Di akhir, script mencetak daftar **unduhan manual** (installer VS Code/Chrome/driver, VSIX, dataset yang butuh login).

**2. Salin folder `bundle` ke SSD eksternal.**

**3. Di PC target, tanpa internet**

```bash
D:\bundle\tools\02_install_offline.bat -Dest C:\gemastik
```

Script memeriksa kelengkapan bundle (menangkap salinan USB yang terpotong), memilih varian torch dari versi CUDA driver (`cu130` → `cu126` → `cpu`, dan turun otomatis kalau GPU tidak terbaca), memasang paket dengan `--no-index`, menyalin model ke disk lokal, lalu menjalankan verifikasi. Setelah selesai: `C:\gemastik\start_jupyter.bat`.

## Isi tambahan untuk tema MBG (profil `plus`)

| Kode | Isi | Untuk apa |
|---|---|---|
| `F6c` | `Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis` | Sentimen pembanding, dipakai pada label TikTok MBG publik; dibutuhkan `label_fingerprint` |
| `H12` | NusaX lengkap (12 bahasa: sentimen, padanan kata, terjemahan) | Kata penanda bahasa daerah |
| `H13` | Korpus Wikipedia 8 bahasa daerah (AksaraLLM) | Eksplorasi kosakata, tanpa label |
| `H14` | Kode wilayah Kepmendagri 2025 | Gazetteer kabupaten/kota, hasil OCR sehingga perlu dicek |
| `H15` | Batas provinsi dan kabupaten (geoBoundaries, CC BY 3.0 IGO) | Peta provinsi untuk slide |
| `H16` | Komentar YouTube MBG (tanpa label) | Latihan |

Paket Python yang ditambahkan: `duckdb`, `ftfy`, `dateparser`, `ruptures`, `yake`, `geopandas`, `indoNLP`.

## Latihan wajib (jangan lewati)

Jalankan langkah 3 di laptop dengan **Wi-Fi mati**, di folder tujuan baru, lalu **reboot** dan jalankan `03_verify_offline.py` lagi. Latihan baru sah kalau lolos setelah reboot. Angka waktu dari verifikasi menggantikan semua estimasi di laporan.

## Yang belum diketahui, tanyakan panitia sebelum 10 November

1. Boleh mencolok SSD eksternal/USB ke PC panitia saat Set-Up Work Station? Kalau tidak, bagaimana cara memasukkan environment?
2. Hak admin ada? Python, VC++ Redistributable, dan driver NVIDIA sudah terpasang? Berapa versi driver (`nvidia-smi`)?
3. Environment yang dipasang saat Set-Up dijamin tetap ada di PC yang sama pada 11 November?
4. Model pre-trained, data eksternal, dan LLM lokal boleh dipakai?
5. Dataset ada di disk lokal PC atau diambil dari server panitia? Format submission dan batas jumlah submission?
6. Slide: bagaimana memindahkan grafik/hasil dari PC ke laptop (data tidak boleh keluar)? File presentasi final paling lambat 21.00 WIB pada tanggal berapa?

## Catatan teknis

- `requirements.in` memakai `onnxruntime` (CPU), bukan `onnxruntime-gpu`, karena build CUDA-nya bisa bentrok dengan DLL CUDA milik torch.
- Wheelhouse hanya untuk Python **3.12** Windows x64. Installer Python 3.12.10 ikut dalam bundle.
- Model `indobert-base-p1` (F2) hanya punya `.bin`. Konversi ke safetensors selagi online jika dipakai (lihat laporan, Langkah 5).
- Jangan pakai data MBG/eksternal sebelum aturan data eksternal jelas.
