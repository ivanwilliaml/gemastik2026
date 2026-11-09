# Notebook cheat sheet

28 notebook, satu per jenis task NLP, ditambah satu notebook setup. Tiap notebook punya struktur yang sama: kartu masalah, kapan dipakai, config, EDA, preprocessing, validasi, baseline, model utama, post-processing, post-analysis, keluaran. Sel yang sudah dijalankan menyimpan outputnya, jadi bisa dibaca langsung di GitHub tanpa menjalankan apa pun.

> Ini versi notebook lengkap. Untuk alur "pilih task, salin sel, jalankan" yang lebih ringkas, pakai [Toolkit](../toolkit/).

## Isi

Empat notebook review pasangan teks dan waktu:

| Notebook | Isi |
|---|---|
| [20 kemiripan review](e_pasangan_teks/20_kemiripan/20_kemiripan_review.ipynb) | STS kontinu, lima pendekatan, group split endpoint, kalibrasi dan error per sumber |
| [21 NLI review](e_pasangan_teks/21_nli/21_nli_review.ipynb) | IndoNLI entailment/neutral/contradiction, lima pendekatan, audit lay vs expert |
| [22 retrieval review](e_pasangan_teks/22_retrieval/22_retrieval_review.ipynb) | TF-IDF, BM25, chunk E5, hybrid, reranker; multi-relevant metrics dan binary matching |
| [02 tren waktu review](a_tanpa_label/02_tren_topik_waktu/02_tren_topik_waktu_review.ipynb) | Empat metode topik, tiga estimasi sentimen, split waktu, volume/coverage dan ketidakpastian |

Tambahan: [12 multilabel review](b_klasifikasi/12_multi_label/12_multi_label_review.ipynb)
berisi lima pendekatan, masked loss/evaluasi untuk unknown labels, iterative stratification
pada grup duplikat, threshold global/per-label cross-fit, ensemble, dan audit hierarki HS.
Sumbernya `_build/build_multilabel_review.py`; konfigurasi run Kaggle tersimpan di
`_build/multilabel_review_kaggle/`. Notebook lama dipertahankan.

Sumber pembangun: `_build/build_pair_time_review.py`. Keempat file berdiri sendiri,
menampilkan EDA angka dan tidak menjalankan grafik. Eksekusi model hanya di Kaggle;
konfigurasi run gabungan tersimpan di `_build/pair_time_kaggle/`.

Unduh hasil yang sudah tersedia dengan `python notebook/_build/sync_pair_time_review.py`
dari folder proyek. Script ini tidak menunggu run yang masih berjalan dan tidak menimpa
sumber dengan notebook yang gagal. Status lengkap per task ada pada `suite_status.json`
di output Kaggle; status run gabungan saja tidak menggantikan audit notebook per task.

Tambahan: [`preprocessing_universal/Preprocessing_universal.ipynb`](preprocessing_universal/Preprocessing_universal.ipynb)
berisi preprocessing **teks saja, langkah demi langkah**: cleaning teknis, normalisasi kata,
cleaning agresif, tokenisasi, audit, dan ekspor. Setiap langkah menampilkan before/after.
Setiap metode memakai satu contoh sintetis independen dan satu tabel before/after.
Run All tidak membaca dataset nyata; data MBG tetap tersedia untuk pemakaian pipeline.

| Folder | Notebook |
|---|---|
| `eda_universal/` | **EDA universal, satu notebook untuk data apa pun.** 21 modul dari skema sampai ablation. Tiap modul menampilkan semua kondisi yang mungkin dan menyorot kondisi data saat ini, lalu merangkumnya jadi laporan keputusan di `eda_out/`. Ubah sel Konfigurasi saja. Kolom yang tidak ada otomatis dilewati |
| `00_setup_offline/` | **Setup offline.** Butuh internet dan cukup dijalankan sekali: mengunduh model, data tambahan, dan wheel library menjadi satu paket, lalu menulis `pack_manifest.json` yang dicari notebook lain |
| `a_tanpa_label/` | 01 topic modeling, 02 tren topik dan sentimen per waktu, 03 clustering, 04 weak supervision dan zero-shot, 05 kata kunci, 06 anomali teks, 07 near-duplicate, spam, dan bot |
| `b_klasifikasi/` | 08 semi-supervised, 09 biner, 10 multi-kelas dengan imbalance, 11 domain shift, 12 multi-label, 13 aspect-based sentiment |
| `c_regresi/` | 14 regresi dan ordinal |
| `d_label_per_token/` | 15 NER, 16 POS tagging, 17 term aspek, 18 keyphrase supervised, 19 QA span |
| `e_pasangan_teks/` | 20 kemiripan dan parafrase, 21 NLI, 22 retrieval dan ranking, 23 konsistensi judul dan isi |
| `f_generatif/` | 24 summarization, 25 terjemahan, 26 normalisasi teks informal, 27 parafrase dan augmentasi, 28 LLM lokal untuk ekstraksi terstruktur |
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
