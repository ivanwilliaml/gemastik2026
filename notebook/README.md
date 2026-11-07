# Notebook cheat sheet

28 notebook, satu per jenis task NLP, ditambah satu notebook setup. Tiap notebook punya struktur yang sama: kartu masalah, kapan dipakai, config, EDA, preprocessing, validasi, baseline, model utama, post-processing, post-analysis, keluaran. Sel yang sudah dijalankan menyimpan outputnya, jadi bisa dibaca langsung di GitHub tanpa menjalankan apa pun.

> Ini versi notebook lengkap. Untuk alur "pilih task, salin sel, jalankan" yang lebih ringkas, pakai [NLP-Competition-Toolkit](../NLP-Competition-Toolkit/).

## Isi

| Folder | Notebook |
|---|---|
| `00_setup_offline/` | **Setup offline.** Butuh internet dan cukup dijalankan sekali: mengunduh model, data tambahan, dan wheel library menjadi satu paket, lalu menulis `pack_manifest.json` yang dicari notebook lain |
| `A_tanpa_label/` | 01 topic modeling, 02 tren topik dan sentimen per waktu, 03 clustering, 04 weak supervision dan zero-shot, 05 kata kunci, 06 anomali teks, 07 near-duplicate, spam, dan bot |
| `B_klasifikasi/` | 08 semi-supervised, 09 biner, 10 multi-kelas dengan imbalance, 11 domain shift, 12 multi-label, 13 aspect-based sentiment |
| `C_regresi/` | 14 regresi dan ordinal |
| `D_label_per_token/` | 15 NER, 16 POS tagging, 17 term aspek, 18 keyphrase supervised, 19 QA span |
| `E_pasangan_teks/` | 20 kemiripan dan parafrase, 21 NLI, 22 retrieval dan ranking, 23 konsistensi judul dan isi |
| `F_generatif/` | 24 summarization, 25 terjemahan, 26 normalisasi teks informal, 27 parafrase dan augmentasi, 28 LLM lokal untuk ekstraksi terstruktur |
| `nlp_kit/` | Library kecil yang dipakai semua notebook: `text` (pembersihan dan preprocessing), `data` (memuat data apa pun ke satu bentuk), `validate` (metrik dan fold), `models` (TF-IDF, embedding, transformer, cross-fit), `decide` (post-processing dan penyimpanan hasil), `core` (lingkungan offline dan pengatur waktu) |
| `_build/` | Generator notebook, bukan untuk dibaca saat lomba (lihat di bawah) |

Setiap folder task berisi `<nomor>_<nama>.ipynb` dan `hasil/ringkasan.json` (ringkasan run). File `hasil/*.csv` (prediksi, OOF) tidak ikut repo karena dibuat ulang saat notebook dijalankan.

## Peringatan data

Beberapa notebook memakai komentar MBG publik yang labelnya **hasil keluaran model**, bukan anotasi manusia. Skornya mengukur kemiripan dengan model itu, bukan kebenaran sentimen. Peringatan ini juga tertulis di notebook yang bersangkutan.

## Cara kerja `_build/`

Sumber notebook ada di `_build/src/<id>.py` dalam format percent (`# %%` untuk sel kode, `# %% [markdown]` untuk teks, `# %% include nama` untuk menyisipkan blok dari `_build/blocks/`). Notebook adalah hasil build, jadi ubah sumbernya, bukan file `.ipynb`.

| Perintah (dari folder `_build/`) | Fungsi |
|---|---|
| `python build.py --mode cepat` | Membangun semua notebook yang sumbernya ada. `--only 08 01` untuk sebagian, `--mode penuh` untuk run penuh |
| `.\run_kaggle.ps1 -Ids 08,01 -Mode cepat -Parallel 2` | Build, push ke Kaggle, tunggu selesai, unduh notebook dan hasil kecil, lalu audit (butuh Kaggle CLI yang sudah login) |
| `python audit.py --only 08` | Memeriksa notebook hasil run: hanya membaca file |
| `python tidy.py jalur/notebook.ipynb` | Membuang metadata eksekusi papermill |
| `python uji_kit.py --shapes D --cpu` | Menguji `nlp_kit` pada beberapa bentuk data |

`tasks.py` memuat daftar notebook, slug kernel, dan dataset Kaggle yang dibutuhkan masing-masing.
