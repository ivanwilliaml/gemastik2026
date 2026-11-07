# GEMASTIK XIX 2026, Data Mining: persiapan NLP offline

Semua bahan persiapan final **GEMASTIK XIX, Kompetisi III (Penambangan Data)**, 11 November 2026: data teks, tanpa internet, dikerjakan di PC panitia (i7-12700, RAM 32 GB, RTX 3070). Temanya diperkirakan MBG (Makan Bergizi Gratis), tetapi task dan datanya belum diumumkan, jadi semua alat dibuat modular untuk banyak jenis task.

Repo ini berisi tiga hal yang saling melengkapi:

1. **Paket offline** untuk memasang seluruh environment di PC tanpa internet.
2. **Toolkit HTML** untuk dibuka saat lomba: pilih task, salin kode per tahap, jalankan.
3. **Notebook** lengkap per jenis task, plus catatan riset.

## Isi repo

```
.
├── NLP-Competition-Toolkit/   Cheat sheet offline (buka index.html) + kode per task
├── persiapan_h1/              Script unduh, pasang, dan verifikasi environment offline
├── notebook/                  28 notebook per jenis task + library nlp_kit
└── docs/                      Ringkasan persiapan dan catatan riset
```

| Folder | Fungsi | Dibaca |
|---|---|---|
| [`NLP-Competition-Toolkit/`](NLP-Competition-Toolkit/) | Aplikasi statis (HTML, CSS, JS murni, tanpa server dan tanpa internet). Isinya 43 task, 20 pipeline kode berurutan, 293 metode dengan kode, 142 snippet, katalog 49 dataset legal, strategi, diagnosis, panduan hardware dan waktu. Kode Python ada di `snippets/` dan `pipelines/` | [README toolkit](NLP-Competition-Toolkit/README.md) |
| [`persiapan_h1/`](persiapan_h1/) | Tiga script: `01_download_all.py` (unduh semua ke satu folder `bundle`), `02_install_offline.ps1/.bat` (pasang di PC target), `03_verify_offline.py` (cek dan ukur kecepatan nyata) | [README persiapan](persiapan_h1/README.md) |
| [`notebook/`](notebook/) | 28 notebook (tanpa label, klasifikasi, regresi, label per token, pasangan teks, generatif), library `nlp_kit`, dan generator notebook | [README notebook](notebook/README.md) |
| [`docs/`](docs/) | `RINGKASAN_PERSIAPAN_GEMASTIK.md` (fakta lomba, hardware, model, strategi 5 jam, risiko, status uji) dan `riset/` (laporan lengkap serta enam catatan mentah) | langsung |

Tidak ikut repo (terlalu besar atau milik pihak lain): bundle offline sekitar 12 GB, dataset latihan, dan PDF buku saku. Cara mendapatkannya ada di bagian [Yang tidak ada di repo](#yang-tidak-ada-di-repo).

## Mulai cepat

### A. Hanya ingin melihat toolkit

Klik dua kali [`NLP-Competition-Toolkit/index.html`](NLP-Competition-Toolkit/index.html). Tidak perlu instalasi.

1. Pilih task di bilah atas, misalnya *Multiclass Classification*. Seluruh halaman menyaring dirinya mengikuti task itu.
2. Buka **Pipeline per Task**. Task menjadi deretan sel: Setup, Muat data, EDA, Cleaning, Preprocessing, Fitur, Validasi, Model, Transformer, Evaluasi, Post-processing, Submission.
3. Ubah hanya sel Setup (nama file dan kolom), salin, jalankan, lanjut ke sel berikutnya. Ada tombol **Salin sel**, **Salin semua sel**, dan **Unduh .ipynb**.

`Ctrl K` atau `/` membuka pencarian. Detail tiap halaman ada di [README toolkit](NLP-Competition-Toolkit/README.md).

### B. Menyiapkan environment untuk lomba

Butuh internet, Python 3.12, dan sekitar 12 GB ruang kosong. Pakai **path pendek** (misalnya `C:\gemastik`): pip gagal menulis beberapa file di path panjang.

```bash
python persiapan_h1/01_download_all.py --out D:\comp\bundle --check
python persiapan_h1/01_download_all.py --out D:\comp\bundle --profile plus
```

`--check` hanya memeriksa link dan ukuran. Profil `core` sekitar 8 GB, `plus` sekitar 12 GB, `all` sekitar 30 GB (termasuk LLM lokal). Di akhir, script mencetak daftar unduhan yang harus manual (installer VS Code dan Chrome, driver NVIDIA, dataset yang butuh login).

Salin folder `bundle` ke SSD eksternal. Di PC target, **tanpa internet**:

```bash
D:\bundle\tools\02_install_offline.bat -Dest C:\gemastik
```

Script memeriksa kelengkapan bundle, memilih varian torch dari versi CUDA driver, memasang paket dengan `--no-index`, menyalin model, lalu menjalankan verifikasi. Setelah itu buka Jupyter lewat `C:\gemastik\start_jupyter.bat`.

**Latihan wajib:** jalankan langkah ini di laptop dengan Wi-Fi mati dan folder tujuan baru, lalu reboot dan jalankan `03_verify_offline.py` lagi. Latihan baru sah kalau lolos setelah reboot.

### C. Menjalankan notebook

Lihat [README notebook](notebook/README.md). Notebook bersifat mandiri per task dan memakai library `nlp_kit` di folder yang sama.

## Rencana hari H

| Tanggal | Kegiatan |
|---|---|
| 10 Nov, 13.00 | Set-Up Work Station: pasang bundle, jalankan verifikasi, catat benchmark RTX 3070 |
| 10 Nov, 16.00 | Technical Meeting: tanyakan aturan yang belum jelas (daftar pertanyaan di `docs/RINGKASAN_PERSIAPAN_GEMASTIK.md`, bagian 9) |
| 11 Nov, 13.00 sampai 18.00 | Lomba 5 jam |
| 12 Nov | Presentasi 20 menit per tim |

Urutan kerja 5 jam dengan gate waktu (T40, T70, T105, ...) ada di halaman *Info Lomba* toolkit dan di bagian 7 ringkasan. Di dalam kode, `clock("tahap")` mencetak menit berjalan dan gate berikutnya, sedangkan `scale_plan` otomatis meringankan eksperimen kalau data melewati 100 ribu baris.

Aturan penting dari panitia: laptop pribadi **tidak boleh menyentuh dataset** (termasuk untuk EDA), dan notebook serta data tidak boleh dibawa pulang. Karena itu seluruh alat di sini dirancang untuk dipasang di PC panitia, dan bahan slide dibuat selagi lomba berjalan.

## Status pengujian

Diukur pada 7 November 2026 di laptop dengan GTX 1650 (CUDA), memakai environment yang dipasang dari bundle tanpa internet (Python 3.12, torch 2.14.1+cu130). Rincian dan cara mengulangnya ada di [`sources.md`](NLP-Competition-Toolkit/references/sources.md).

| Uji | Hasil |
|---|---|
| Tiap snippet dijalankan sendiri di namespace kosong, dengan model bundle | 142 dari 142 lolos |
| Semua sel tiap task dijalankan berurutan dalam satu proses, tanpa transformer | 22 dari 22 skenario lolos, termasuk klasifikasi pada komentar TikTok MBG asli yang kotor (duplikat, kosong, label `empty`) dan pipeline ekstraksi |
| Sama, dengan fine-tuning transformer sungguhan | 8 dari 8 lolos (klasifikasi multikelas, biner, MBG, multilabel, regresi, pasangan, NER, QA) |

Uji pada data MBG kotor menemukan satu bug nyata (split waktu hanya memvalidasi sebagian baris sehingga evaluasi gagal karena NaN), yang sudah diperbaiki.

**Belum diuji:**

- RTX 3070 dan PC panitia. Jalankan ulang saat Set-Up: `python NLP-Competition-Toolkit/tools/test_pipelines.py`.
- Skala 500 ribu baris (waktu dan RAM). Uji memakai ratusan sampai ribuan baris; `scale_plan` hanyalah pagar, ambangnya belum diukur.
- Model T5 untuk ringkasan (hanya ada di profil `all`) dan fungsi seq2seq.
- Script `02_install_offline` itu sendiri (yang diuji hanya pemasangan wheelhouse lewat pip).
- Notebook di `notebook/` tidak dijalankan ulang dalam sesi ini, dan tombol salin di browser belum diuji dengan clipboard asli.

Skor dari sampel kecil tidak bermakna: uji hanya membuktikan kode berjalan. Prioritas task dan probabilitasnya adalah penilaian, bukan informasi panitia.

## Yang tidak ada di repo

| Apa | Kenapa | Cara mendapatkan |
|---|---|---|
| Bundle offline (`bundle/`, sekitar 12 GB) | Terlalu besar | `python persiapan_h1/01_download_all.py --out D:\comp\bundle --profile plus` |
| Dataset latihan (`NLP-Competition-Toolkit/datasets/<key>/`) | Besar dan milik pihak lain | `python NLP-Competition-Toolkit/tools/download_dataset.py --list`, lalu sebut key-nya. Katalog dan lisensi: `datasets/CATALOG.md` |
| `Buku Saku Gemastik XIX - 2026 portrait.pdf` | Dokumen resmi panitia | Unduh dari situs GEMASTIK |
| `notebook/**/hasil/*.csv` | Dibuat ulang saat notebook dijalankan | Jalankan notebook-nya |

Setelah mengunduh dataset, jalankan `python NLP-Competition-Toolkit/tools/build_library.py` supaya halaman Dataset menampilkan status "tersimpan lokal" dengan benar.

## Catatan lisensi dan data

- Dataset di katalog dan di bundle punya lisensi masing-masing (sebagian CC BY-NC-SA). Pakai untuk latihan; saat lomba, gunakan data eksternal hanya bila panitia mengizinkan.
- Data MBG publik yang dipakai untuk latihan sebagian besar berlabel otomatis (bukan anotasi manusia). Jangan menafsirkan skornya sebagai kebenaran sentimen.
- Leksikon aspek MBG, probabilitas task, dan angka ambang di dokumen adalah penilaian (hipotesis), bukan informasi dari panitia.
