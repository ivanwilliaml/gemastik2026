# NLP Competition Toolkit

Cheat sheet offline untuk final Data Mining GEMASTIK XIX (11 November 2026). Buka `index.html` di browser dengan klik dua kali. Tidak perlu server, internet, atau instalasi.

## Cara pakai saat lomba

1. Pilih task di bilah atas, misalnya **Multiclass Classification**.
2. Buka **Pipeline per Task**. Task itu menjadi deretan sel: Setup, Muat data, EDA, Cleaning, Preprocessing, Fitur, Validasi, Model, Transformer, Evaluasi, Post-processing, Submission.
3. Salin sel Setup, ubah nama file dan kolom, jalankan. Lanjut ke sel berikutnya. Variabel nyambung antar sel, dan fungsi yang dibutuhkan sudah ada di dalam selnya.

Tombol **Salin semua sel** dan **Unduh .ipynb** tersedia di halaman yang sama.

## Mix and match

Tiap tahap punya satu fungsi utuh. Ganti namanya di sel Setup:

| Fungsi | Pilihan |
|---|---|
| `preprocess(t, preset)` | `none`, `tfidf`, `social`, `topic`, `transformer`, `indobertweet`. Ubah satu langkah: `preprocess(t, "tfidf", emoji="remove", email="token")` |
| `eda_report(train, test, text, target)` | Profil, kosong, duplikat, panjang, target, kata khas, OOV, overlap, adversarial, saran |
| `clean_dataset(df, text, target)` | Kosong, pseudo-null, dedup, near-duplicate, label bertentangan, grup fold |
| `make_vectorizer(kind)` | `tfidf_union`, `tfidf_word`, `tfidf_char`, `count`, `binary`, `hashing`, `lsa`, `meta`, `tfidf_meta`, `w2v`, `w2v_idf`, `fasttext`, `doc2vec`, `embed`, `raw` |
| `make_folds(y, groups, times, kind)` | `auto`, `stratified`, `kfold`, `regression`, `group`, `strat_group`, `multilabel`, `time`, `source` |
| `make_model(name)` | `lr`, `svc`, `svc_cal`, `sgd`, `ridge_clf`, `cnb`, `mnb`, `lgbm`, `xgb`, `rf`, `mlp`, `smote_lr`, `dummy`, `cnn`, `gru`, `bigru`, `lstm`, `bilstm`, `attn`, `ovr`, `chain`, `ridge`, `ridge_log`, `lgbm_reg`, `lgbm_l1`, `rf_reg`, `dummy_reg` |
| `oof_predict(make_vec, make_mdl, texts, y, folds, test_texts)` | Menyatukan semuanya, vectorizer di-fit di dalam fold |

Contoh di sel Setup klasifikasi:

```python
EXPERIMENTS = [("tfidf_union", "lr"), ("tfidf_char", "svc"), ("lsa", "lgbm"), ("raw", "bigru")]
```

Model neural (`cnn`, `gru`, dan seterusnya) harus berpasangan dengan vectorizer `raw`. `cnb` tidak bisa memakai fitur negatif seperti `lsa`.

## Domain MBG

Folder `snippets/domain/` berisi alat khusus tema MBG. Semuanya tanpa model besar dan jalan di CPU.

| Fungsi | Untuk apa |
|---|---|
| `tag_terms(texts)`, `term_hits`, `audit_aspects`, `aspect_sentiment` | Kamus 13 aspek MBG (keamanan pangan, anggaran, korupsi, dapur, dst.), multi-label dengan Aho-Corasick. `term_hits` membuang istilah nol hit, `audit_aspects` mengambil 30 contoh per aspek untuk dibaca |
| `language_report(texts)`, `detect_local_language` | Porsi 10 bahasa daerah (Jawa, Sunda, Minang, Bali, Banjar, Madura, Aceh, Bugis, Batak Toba, Ngaju) dengan kata penanda dari NusaX. Menampilkan batas ketat dan longgar |
| `label_fingerprint(texts, labels, MODELS)` | Melacak apakah label dibuat oleh salah satu model bundle atau InSet. Kecocokan di atas 98% berarti label hampir pasti buatan model itu |
| `extract_records(texts, times)`, `parse_money`, `parse_quantity`, `parse_dates` | Uang (Rp10.000, 15rb, 174 triliun), jumlah (3 siswa, ratusan murid), tanggal (kemarin, 24/09/2025), makanan, gejala, insiden, entitas |
| `tag_locations(texts)`, `resolve_location(name)`, `plot_province_map` | 514 kabupaten/kota dan 38 provinsi (Kepmendagri 2025), nama bertypo lewat RapidFuzz, peta provinsi untuk slide |
| `issue_timeline`, `change_points`, `share_over_time`, `plot_issue_timeline` | Proporsi isu per periode (bukan hanya jumlah), titik perubahan dengan `ruptures`, garis kejadian dari `MBG_EVENTS` |

Pipeline **Ekstraksi Informasi & Peta Isu MBG** merangkai semuanya dari data mentah sampai `bahan_slide.pptx`, tanpa label dan tanpa GPU. Pipeline klasifikasi juga punya sel *Lacak sumber label* (`CHECK_LABEL`) dan *Bahan slide MBG* (`MBG_INSIGHT`).

Catatan: sebutan lokasi bukan bukti lokasi kejadian, angka dari komentar bukan data resmi, dan kamus adalah hipotesis. Audit presisi pada 30 sampel sebelum dipakai di slide.

## Jam lomba dan skala

`start_clock("13:00")` dan `clock("tahap")` mencetak menit berjalan dan gate berikutnya (T40, T70, T105, T150, T225, T240, T280, T300). `scale_plan` mengecilkan eksperimen otomatis: di atas 100 ribu baris fitur char dibuang dan fold jadi 3, di atas 250 ribu hanya `hashing + sgd`. `scale_plan(..., force=True)` mematikan pagar. Ambang ini belum diukur di PC panitia.

## Filter task

Task terpilih menyaring semua halaman: metode, metrik, strategi, diagnosis, snippet, model, notebook, dataset, dan pencarian. **Tanpa pretrained** menyembunyikan metode yang butuh model pretrained.

## Halaman

| Halaman | Kegunaan |
|---|---|
| Beranda, Info Lomba | Checklist 25 menit, rencana 5 jam, aturan, jadwal, pertanyaan ke panitia |
| Decision Guide | Wizard, pohon keputusan, tabel cepat |
| Pipeline per Task | Sel kode per tahap untuk 43 task |
| Metode per tahap | 293 metode, tiap kartu menampilkan kodenya |
| Keranjang Kode | Kumpulkan snippet terpilih, dependensi ikut otomatis |
| Snippet Library, Kamus Metrik, Model | Fungsi satuan, rumus, item model bundle |
| Strategy, Diagnostics, Domain MBG | Solusi per kondisi, leksikon aspek |
| Hardware, Versi & Offline, Library, Dataset, Progres | Persiapan |

`Ctrl K` atau `/` membuka pencarian.

## Menguji di PC panitia

Jalankan setelah bundle terpasang. Perintah pertama menguji semua snippet, yang kedua semua pipeline berurutan.

```bash
python tools/test_snippets.py --models C:/gemastik/models
```

```bash
python tools/test_pipelines.py
```

```bash
python tools/test_pipelines.py classification --bert --comp-root C:/gemastik
```

Pengujian butuh dataset latihan di `datasets/<key>/`. Folder itu tidak ikut repo; unduh dulu (butuh internet) lalu bangun ulang:

## Menambah atau mengubah kode

- Snippet: `snippets/<kategori>/<nama>.py`. Header opsional `# title:`, `# when:`, `# avoid:`, `# tags:`, `# needs:`.
- Pipeline: `pipelines/<nama>.py`. Tiap `# %% stage: id | Nama` menjadi satu sel, `# uses:` menyebut snippet yang dimasukkan.
- Notebook sendiri: `notebooks/<kategori>/<task_id>__<varian>.ipynb`, atau daftarkan di `notebooks/manifest.json`.

Setelah mengubah, jalankan `python tools/build_library.py` lalu muat ulang browser.

## Dataset

```bash
python tools/download_dataset.py --list
python tools/download_dataset.py indonlu_sentiment sms_spam
python tools/build_library.py
```

Hasil unduhan masuk ke `datasets/<key>/`. Ini untuk latihan. Saat lomba, pakai hanya bila panitia mengizinkan data eksternal.
