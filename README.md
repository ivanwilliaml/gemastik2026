# GEMASTIK XIX 2026 · Persiapan Data Mining / NLP offline

Tempat kerja untuk notebook, toolkit lomba, paket instalasi offline, dan dokumen persiapan.
Tema MBG dipakai untuk latihan; jenis task final dan aturan yang belum pasti dijelaskan di dokumen riset.

## Mulai dari sini

| Kebutuhan | Buka |
|---|---|
| Preprocessing komentar MBG, langkah demi langkah | [Notebook preprocessing](notebook/preprocessing_universal/Preprocessing_universal.ipynb) |
| Panduan memakai dan mengubah preprocessing | [README preprocessing](notebook/preprocessing_universal/README.md) |
| Eksplorasi dataset baru | [EDA universal](notebook/eda_universal/EDA_universal.ipynb) |
| Pilih pipeline NLP saat lomba | [Toolkit offline](toolkit/index.html) |
| Siapkan instalasi PC tanpa internet | [Panduan persiapan offline](persiapan_offline/README.md) |
| Aturan, strategi, hardware, dan riset | [Daftar dokumen](docs/README.md) |

## Susunan folder

```text
Gemastik/
├── notebook/             Notebook utama, notebook per task, dan library nlp_kit
├── toolkit/              Aplikasi HTML offline, pipeline, snippet, dataset latihan
├── persiapan_offline/    Script download, instalasi offline, dan verifikasi
├── docs/                 Buku saku, ringkasan persiapan, dan riset
├── .gitignore
└── README.md
```

Nama seluruh folder memakai huruf kecil dengan pemisah garis bawah (`snake_case`).
Masing-masing bagian mempunyai README sendiri. Buka toolkit/index.html langsung dari browser;
tidak memerlukan server. Notebook per task tetap memakai struktur A–F dan library yang sama.

## Preprocessing MBG

Notebook utama memakai **32 contoh sintetis: satu metode, satu contoh before/after**.
Contohnya independen agar perubahan dari setiap metode terlihat jelas. Bagian akhir menunjukkan
API sekali panggil dan kode opsional untuk dataset baru.

Data asli tetap ada di `notebook/preprocessing_universal/data/MBG_PROCESSED.csv` (57.362 komentar).
Versi notebook dengan output data nyata tersimpan di
[arsip data nyata](notebook/preprocessing_universal/legacy/Preprocessing_mbg_real.ipynb).
Ekspor data nyata di `hasil_stepwise/` tetap tersedia dan tidak ditimpa demo.
Label pada CSV adalah keluaran model, bukan anotasi gold.

Pada folder preprocessing: `_build/` berisi pembangun notebook, `tests/` pengujian,
`reports/` bukti verifikasi, dan `legacy/` versi terdahulu.

## Menyiapkan paket offline

Dari folder utama, saat ada internet:

```powershell
python persiapan_offline/01_download_all.py --out D:\comp\bundle --check
python persiapan_offline/01_download_all.py --out D:\comp\bundle --profile plus
```

Panduan instalasi, kebutuhan Python/Windows, latihan tanpa Wi-Fi, dan verifikasi GPU ada di
[persiapan_offline/README.md](persiapan_offline/README.md). Bundle, model, wheelhouse, dan dataset
lokal tidak dimasukkan ke Git; menyimpan repo saja belum cukup untuk memindahkan seluruh environment.

## Mengubah isi proyek

- **Preprocessing:** ubah `stepwise.py` / `lexicons.py` dan sumber dalam `_build/`, lalu rebuild notebook.
- **Notebook task:** ubah sumber `notebook/_build/src/` dan blok bersama, kemudian build ulang.
- **Toolkit:** ubah `toolkit/snippets/` atau `toolkit/pipelines/`, lalu jalankan `python toolkit/tools/build_library.py`.
- **Data asli:** pertahankan sumbernya; hasil turunan disimpan di folder hasil.

Pengujian toolkit terdahulu tersimpan di [toolkit/references/](toolkit/references/).
Validasi di PC panitia, skala 500 ribu baris, dan peningkatan skor task tetap memerlukan pengukuran tersendiri.
Catatan penataan terbaru tersedia di [docs/CATATAN_PERAPIAN.md](docs/CATATAN_PERAPIAN.md).
