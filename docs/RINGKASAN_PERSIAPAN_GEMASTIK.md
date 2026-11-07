# Ringkasan induk: persiapan final GEMASTIK XIX, Data Mining (NLP, offline, MBG)

Dibuat 6 November 2026. File ini merangkum semua yang diketahui sampai sekarang: fakta lomba, hardware, model, task yang mungkin, strategi, environment, domain MBG, risiko, dan status pengujian. Detail penuh dan sumber ada di `riset/Laporan_lengkap_Persiapan_final_NLP_offline_MBG.md` (sekitar 8.600 kata) dan enam catatan mentah di `riset/catatan_mentah/`.

**Label kepastian yang dipakai:** **[Panitia]** = tertulis dari panitia; **[Buku saku]** = dari buku saku PDF; **[Ukur]** = diukur langsung di laptop Ivan; **[Terbit]** = dari paper/kartu model; **[Estimasi]** = perkiraan, belum diukur; **[Penilaian]** = pendapat, bukan data; **[Belum teruji]** = belum pernah dijalankan.

---

## 1. Intisari

1. Lomba ini hampir pasti **GEMASTIK XIX, Kompetisi III (Penambangan Data)**. Pesan panitia untuk "Peserta Divisi III – Data Mining" cocok dengan deskripsi: data teks, tanpa internet. Tema MBG dan 500.000 baris berasal dari Ivan, bukan dari dokumen publik.
2. **Pengerjaan di PC panitia, bukan laptop Ivan**: i7-12700, RAM 32 GB DDR5, RTX 3070 LHR, SSD 1 TB. Laptop pribadi **dilarang menyentuh dataset**. Laptop hanya untuk eksplorasi terpisah, dokumentasi, dan slide.
3. Notebook dan data **tidak boleh dibawa pulang**. Konten slide harus sudah masuk sebelum lomba selesai. File presentasi final paling lambat **21.00 WIB**.
4. Karena itu seluruh persiapan berpusat pada **paket offline yang portabel**: satu folder `bundle` di SSD eksternal, dipasang di PC panitia saat Set-Up Work Station (10 November).
5. Tiga script sudah dibuat dan diuji sebagian. Yang belum teruji: build wheelhouse penuh, torch CUDA, dan benchmark di RTX 3070.
6. Pertanyaan terpenting ke panitia: **boleh mencolok SSD/USB ke PC?** Tanpa itu script tidak punya jalur masuk.

---

## 2. Fakta lomba

### 2.1 Aturan dari panitia [Panitia]

- Data yang diproses berupa teks.
- Perlombaan tanpa akses internet.
- Tiap tim mendapat 1 PC: Intel Core i7-12700, RAM 32 GB DDR5 4800 UDIMM (2×16 GB), SSD 1 TB 2280 PCIe NVMe, GPU NVIDIA GeForce RTX 3070 LHR.
- Akses dataset, training model, dan submission hanya lewat PC panitia.
- Tiap anggota boleh membawa laptop pribadi, tetapi akses dataset/training/submission wajib di PC panitia. Jawaban panitia ke pertanyaan peserta: **laptop strict tidak boleh mengakses dataset**, dipakai untuk eksplorasi terpisah. Bahkan EDA di laptop tidak boleh.
- FAQ-01: konten penting presentasi harus sudah masuk slide sebelum kompetisi berakhir. File presentasi final dikumpulkan paling lambat 21.00 WIB. **Tanggal tenggat itu belum dikonfirmasi.**
- FAQ-02: data dan notebook utama tetap di PC panitia, tidak boleh dibawa pulang saat finalisasi slide.
- Ada perbedaan istilah: FAQ menyebut "kompetisi Kaggle", jadi platform lomba kemungkinan mirip Kaggle (leaderboard + submission). Cara submission tanpa internet belum diketahui.

### 2.2 Jadwal [Buku saku, dibaca dari urutan teks hasil ekstraksi PDF, cek ulang di PDF]

| Tanggal | Waktu (WIB) | Kegiatan | Tempat |
|---|---|---|---|
| Selasa 10 Nov | 13.00–16.00 | **Set-Up Work Station** Kompetisi III | B0201 (Lantai 2) |
| Selasa 10 Nov | 16.00–17.00 | Technical Meeting Kompetisi III | B0201 |
| Rabu 11 Nov | 13.00–18.00 | **Penyelesaian kasus** (5 jam) | B0201 |
| Kamis 12 Nov | sekitar 08.00–17.00 | Presentasi solusi, **20 menit per tim** | A0205 (Lantai 2) |
| Jumat 13 Nov | 19.00–21.00 | Penutupan dan pengumuman | Sport Hall |

**Konflik sumber:** panduan GEMASTIK 2026 menyebut set-up 10 Nov pukul 12.00–18.00; buku saku menyebut 13.00–16.00. Dipakai versi buku saku karena lebih spesifik. Konfirmasi di Technical Meeting.

### 2.3 Bobot nilai [Terbit: Panduan GEMASTIK 2026, hal. 27]

Babak penyisihan 20%, skor leaderboard 30%, ketepatan metode 25%, presentasi 25%. Artinya 50% nilai ada di metode dan presentasi. Dedup yang jujur, validasi yang benar, dan rekomendasi yang jelas sama pentingnya dengan selisih kecil di leaderboard.

### 2.4 Preseden [Terbit: ITB]

Pada GEMASTIK XIV, penyisihan berupa klasifikasi tweet vaksin, tetapi **final berubah menjadi question answering**. Tim perunggu ITB juga baru menemukan selisih besar antara skor internal dan skor resmi 10–30 menit sebelum tenggat. Pelajaran: siapkan cabang task cadangan dan submit lebih awal.

### 2.5 Yang belum diketahui

Pretrained model, data eksternal, dan LLM lokal boleh atau tidak. Boleh atau tidak mencolok USB/SSD. Hak admin. Versi driver NVIDIA. Python dan VC++ sudah terpasang atau belum. Apakah environment yang dipasang tanggal 10 tetap ada tanggal 11. Format dan batas submission. Skema dataset dan jenis label.

---

## 3. Hardware

| | PC panitia (target hari H) | Laptop Ivan (hanya persiapan) |
|---|---|---|
| CPU | i7-12700 (12 core, 20 thread, spesifikasi umum) | i7-9750H 6c/12t |
| RAM | 32 GB DDR5 | 7,8 GB (sering tersisa ~1,4 GB karena app lain) |
| GPU | RTX 3070 LHR (umumnya 8 GB VRAM, bukan dari panitia) | GTX 1650 4 GB |
| Presisi | **bf16 dan fp16 didukung** (Ampere) | fp16 berisiko NaN; tanpa bf16 |
| Disk | NVMe 1 TB | C: 33 GB bebas, D: 283 GB, E: 414 GB |
| Python | belum diketahui | 3.12.6 global (torch 2.12.1+cpu, tanpa CUDA) |

**Dampak:** estimasi waktu di laporan lama dibuat untuk GTX 1650 dan **tidak berlaku** untuk RTX 3070. Saran khusus laptop (fp32 saja, upgrade RAM, page file, LLM di 4 GB) gugur. Dengan 8 GB VRAM dan bf16, batch lebih besar dan fine-tuning jauh lebih cepat, tetapi angka pastinya baru ada setelah `03_verify_offline.py` dijalankan di PC itu.

**Catatan laptop:** saat dicek, RAM terpakai ~83%. Penyebab terbesar: aplikasi Claude (~2,2 GB), Chrome (~1 GB), layanan Windows. Tidak ada training atau unduhan berjalan. Untuk latihan, tutup semua aplikasi lain.

---

## 4. Task yang mungkin

Probabilitas di bawah adalah **[Penilaian]**, bukan data. Tidak ada arsip kasus final resmi yang ditemukan.

| Task | P | Metrik mungkin | Yang dilakukan |
|---|---|---|---|
| Sentimen / stance 3 kelas | 35% | Macro-F1 | TF-IDF word+char, decision layer, IndoBERTweet pada sampel, blend |
| Aspek / topik (multi-class/label) | 20% | Macro/micro-F1 | Binary relevance, threshold per label, leksikon aspek sebagai fitur |
| ABSA | 10% | Macro-F1 pasangan | Input `[aspek] [SEP] teks`, atau model per aspek |
| Deteksi biner (hoaks/relevansi/spam/buzzer) | 10% | F1 kelas positif, PR-AUC | Threshold dari kurva OOF dihaluskan; fitur burst dan near-duplicate |
| QA / NER / ringkasan / kata kunci | 10% | EM, entity-F1, ROUGE | Baseline ekstraktif dulu, lalu `Rifky/Indobert-QA` atau `cahya/NusaBert-ner-v1.3` |
| Emosi / toksisitas | 5% | Macro-F1 | Pipeline sentimen + model siap pakai |
| Regresi engagement | 5% | RMSE/RMSLE | Target `log1p`, TF-IDF SVD + fitur waktu + LightGBM |
| Tanpa label, hanya laporan | 5% | Dinilai juri | Embedding, topik sampel, garis waktu, rekomendasi |

Pengubah yang bisa menempel di task apa pun [Penilaian]: label buatan mesin/berisik 40%, label jarang (<10% baris) 35%, split test berdasarkan waktu/sumber 30%, bagian topik/lonjakan di slide 70%. Alasan: melabeli 500.000 baris manual tidak masuk akal. **Tes cepat:** latih model dangkal pada kata kunci dan bandingkan dengan label panitia; kalau cocok hampir sempurna, label kemungkinan dibuat aturan atau model.

### Pohon keputusan di gate T40

- **Pretrained dilarang:** TF-IDF + LightGBM + word2vec, tanpa BERT.
- **Label banyak (≥ 20.000 setelah dedup):** TF-IDF semua baris, decision layer, IndoBERTweet pada sampel, lalu blend.
- **Label sedikit (±50–5.000):** pisahkan gold; bandingkan TF-IDF dan LR di atas embedding; SetFit hanya kalau mengalahkan LR; aturan kata kunci + zero-shot pada sampel; self-training 1–2 putaran; validasi hanya pada gold.
- **Tanpa label:** dedup, embedding, topik, sentimen/aspek lemah dinilai pada 150–300 gold buatan sendiri, garis waktu, rekomendasi.
- **Pasangan teks / relasional:** TF-IDF biasa tidak cukup (IF26: macro-F1 0,513).

---

## 5. Model

Tidak ada model yang punya skor terbit pada data MBG. Skor di bawah dari domain lain [Terbit].

| Task | Pilihan 1 | Cadangan | Bukti |
|---|---|---|---|
| Fine-tune teks sosial | `indolem/indobertweet-base-uncased` | `LazarusNLP/NusaBERT-base` | Rata-rata 86,1–86,5 di 7 dataset Twitter vs IndoBERT 83,4, mBERT 79,6 |
| Campur Jawa/Sunda | `LazarusNLP/NusaBERT-base` | IndoBERTweet | NusaX jav/sun 84,1/85,2 vs IndoBERT 81,7/77,5 |
| Teks berita | `indobenchmark/indobert-base-p1` | NusaBERT-base | SmSA 87,73; EmoT 75,48 |
| Sentimen siap pakai | `w11wo/indonesian-roberta-base-sentiment-classifier` | `crypter70/IndoBERT-Sentiment-Analysis` | Test macro-F1 91,02 pada SmSA |
| Emosi | `StevenLimcorn/indonesian-roberta-base-emotion-classifier` | `yogie27/IndoRoBERTa-Base-Emotion-Indonesian-Social-Media` | Acc 0,725, F1 0,7336 (log validasi EmoT) |
| Hate speech | `Exqrch/IndoBERTweet-HateSpeech` | `OwLim/indonesian-roberta-hate-speech-detection` | Acc 0,89, macro-F1 0,78 (10-fold) |
| NER | `cahya/NusaBert-ner-v1.3` | `w11wo/indonesian-roberta-base-nerp-tagger` | F1 0,8171 (NERGrit) |
| Zero-shot | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta` | mDeBERTa tanpa skor Indonesia; cadangan 74,69 |
| QA | `Rifky/Indobert-QA` | `timpal0l/mdeberta-v3-base-squad2` | EM 51,61, F1 69,09 (SQuAD terjemahan) |
| Ringkasan | `panggi/t5-small-indonesian-summarization-cased` | ekstraktif | Tanpa skor di kartu |
| Embedding | `intfloat/multilingual-e5-small` | `LazarusNLP/all-indo-e5-small-v4` | MIRACL 74,80; SmSA 83,6 |
| Hoaks | **tidak ada model yang layak** | fine-tune kalau ada label | `theonegareth/IndoHoaxDetector`: 99,71% di split sendiri tapi **ROC-AUC 0,50** pada benchmark independen |
| ABSA | tidak ada | zero-shot pada sampel lalu latih classifier | Hanya repo dengan kartu minim |

**Paket inti yang dibawa (ungated, MIT/Apache 2.0):** `intfloat/multilingual-e5-small` (471 MB), `indolem/indobertweet-base-uncased` (445 MB), `w11wo/indonesian-roberta-base-sentiment-classifier` (499 MB), `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (558 MB), `cahya/NusaBert-ner-v1.3` (644 MB).

**Catatan model:**
- IndoBERTweet menuntut preprocessing yang sama dengan pelatihannya: huruf kecil, mention jadi `@USER`, URL jadi `HTTPURL`, emoji jadi teks.
- `indobert-lite` menghemat disk dan VRAM, tetapi tidak menghemat waktu (hidden size dan jumlah layer sama dengan base).
- **Jangan dipakai:** `google/embeddinggemma-300m` (gated, lisensi Gemma), `jinaai/jina-embeddings-v3` dan `tabularisai/multilingual-sentiment-analysis` (non-komersial), BGE-M3 dan e5-large (terlalu lambat untuk 500k baris di GPU kecil, estimasi 2–4,6 jam di GTX 1650), fastText `.bin` (7,2 GB terdekompresi; pakai `cc.id.300.vec.gz` 1,2 GB dengan batas kata, atau `bstds/ft_id_small_300` 14 MB).
- Enam ID tidak bisa diakses anonim, anggap tidak tersedia: `w11wo/indonesian-roberta-base-bapos-tagger`, `w11wo/indonesian-roberta-base-prdner-tagger`, `ayameRushia/indobert-base-p1-finetuned-indonlu-smsa`, `Exqrch/IndoBERTweet-Toxicity`, `qanastek/pos-indonesian`, `Wikidepia/IndoT5-small-paraphrase`. `cahya/roberta-base-indonesian` hanya ada dengan akhiran `-522M` atau `-1.5G`.
- Beberapa model hanya punya `.bin` (mis. `indobenchmark/indobert-base-p1`). Dalam uji lokal, `transformers` di laptop memuat `.bin` IndoBERTweet tanpa masalah, tetapi versi di wheelhouse (transformers 5.22.1) belum diuji untuk itu. Kalau ragu, konversi ke safetensors selagi online.

### BERT vs TF-IDF: rentangnya lebar

| Data | Klasik | Transformer | Catatan |
|---|---|---|---|
| NusaX Indonesia, macro-F1 | SVM 78,7 | IndoBERT base 90,0 | ≈ +11 |
| SmSA, macro-F1 | fastText 82,13 | Base 87,73 | +5,6 (pembanding bukan TF-IDF) |
| Tweet MBG 54.105 baris, **akurasi** | SVM 83,39 | IndoBERT 87,60 | +4,2; sumber label tidak disebut |
| TikTok MBG, akurasi | SVM 0,73 | IndoBERT 0,78 | +5; hanya dari cuplikan |
| Ulasan produk, macro-F1 | LinearSVC 0,5510 | IndoBERT 0,5088 | −4,2; hanya dari cuplikan |

**Keputusan [Penilaian]:** TF-IDF dulu sebagai submission aman (menit, murah), BERT sebagai taruhan GPU (bisa gagal). Keduanya mengisi blend. Ekspektasi realistis pada studi MBG: macro-F1 0,70–0,87 pada label bersih, sering dari test kecil.

### LLM lokal (hanya kalau aturan mengizinkan)

LLM bukan pengganti NLI/encoder. Gunanya: menamai klaster topik dan melabeli ~1.000 baris sampel. **Jangan pernah dipakai untuk melabeli semua 500.000 baris** (≈139 jam pada 1 detik per baris).

| Fungsi | File | Repo | Ukuran |
|---|---|---|---|
| Label/penamaan topik | `gemma-4-E2B-it-Q4_K_M.gguf` | `unsloth/gemma-4-E2B-it-GGUF` | 3,11 GB |
| Pembanding | `Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf` | `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF` | 3,43 GB |
| Asisten kode | `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` | 2,10 GB |

Skor SEA-HELM Indonesia (leaderboard 18 Sep 2026, kolom tanpa nama diasumsikan skor keseluruhan): Gemma 4 E2B 64,13; SEA-LION v4.5 E2B 63,96. Runtime: `llama-server` build `b####` (mis. `llama-b12119-bin-win-cuda-12.4-x64.zip` + `cudart`). Rilis yang ditandai "latest" (v0.9.0) tidak punya binary Windows. Hindari tag Ollama `gemma4:e2b` (satu laporan: 7,8 GB memori). Estimasi untuk GTX 1650: 500–1.000 baris per 30 menit; **di RTX 3070 belum diukur**. LLM tidak boleh berjalan bersamaan dengan training GPU. Cara JSON-schema output didukung `llama-server`; `-c`, `-t`, `--port` perlu dicek dengan `--help`.

---

## 6. Environment dan script

### 6.1 Lokasi

`D:\Lomba\2026\Gemastik\persiapan_h1\`

| File | Dijalankan di | Internet | Fungsi |
|---|---|---|---|
| `01_download_all.py` | Laptop | Perlu | Unduh wheel Python, model, leksikon, data kecil, docs ke folder `bundle`. Hanya library bawaan Python 3.12. Bisa dilanjutkan, ada verifikasi sha256 dan ukuran |
| `02_install_offline.ps1` / `.bat` | PC panitia (dan latihan di laptop) | Tidak | Cek kelengkapan bundle, pasang Python 3.12.10 per-pengguna kalau perlu, buat venv, pasang paket `--no-index`, pilih torch otomatis, salin model, buat `env.ps1`/`env.bat`/`start_jupyter.bat`, jalankan verifikasi |
| `03_verify_offline.py` | Otomatis dari langkah 2 | Tidak | Cek import, GPU, presisi, dan **ukur kecepatan nyata** (embedding, inferensi, fine-tuning) |
| `requirements.in`, `constraints_cu130/cu126/cpu.txt`, `README.md` | | | Daftar paket dan constraint torch |

### 6.2 Cara pakai

```bash
python 01_download_all.py --out D:\comp\bundle --check
python 01_download_all.py --out D:\comp\bundle --profile plus
```

Profil: `core` (≈8 GB), `plus` (≈12 GB), `all` (≈30,5 GB, termasuk LLM dan fastText). `--only F4,D1` untuk item tertentu. Lalu salin `bundle` ke SSD eksternal, dan di PC target:

```bash
D:\bundle\tools\02_install_offline.bat -Dest C:\gemastik
```

Pemilihan torch otomatis dari "CUDA Version" di `nvidia-smi`: CUDA ≥ 13 → `cu130`; CUDA 12.x → `cu126`; lainnya atau tanpa NVIDIA → `cpu`. Kalau GPU tidak terbaca setelah pemasangan, script mencoba varian lain secara otomatis.

### 6.3 Isi bundle (kode item mengikuti laporan lengkap)

- **Installer:** Python 3.12.10, VC++ Redistributable.
- **Wheelhouse:** `wheelhouse` (torch 2.14.1+cu130 + semua paket), `wheelhouse_cu126`, `wheelhouse_cpu`.
- **Model inti (C1–C5)**, model tambahan (F1–F15), LLM (G, hanya profil `all`).
- **Leksikon dan data kecil:** kamus slang, singkatan, kamus alay, stopword + punkt_tab NLTK, SmSA, EmoT, hate speech multi-label, MBG TikTok berlabel otomatis, fakta MBG, IndoLEM, NusaX, IndoNLI, InSet, sentimen emoji, MBG X berlabel manual (~1,1k baris).
- **Docs offline:** scikit-learn, pandas, NumPy, Matplotlib, Python.
- **Manual (script mencetak daftarnya):** installer VS Code, Chrome (untuk kaleido), driver NVIDIA RTX 3070; docs Transformers; Zeal; VSIX VS Code; ekstensi Continue; dataset Kaggle/Mendeley yang butuh login.

### 6.4 Paket Python

`pandas polars pyarrow numpy scipy scikit-learn lightgbm xgboost transformers datasets accelerate tokenizers sentence-transformers evaluate setfit bertopic umap-learn hdbscan gensim onnxruntime cleanlab optuna matplotlib seaborn plotly kaleido wordcloud networkx jupyterlab ipykernel ipywidgets tqdm joblib pyahocorasick rapidfuzz datasketch emoji fast-langdetect PySastrawi openpyxl python-pptx nltk sentencepiece tiktoken protobuf hf-xet huggingface-hub safetensors psutil nvidia-ml-py`

**Perubahan dari daftar riset awal:** `onnxruntime-gpu` diganti `onnxruntime` (build CUDA 12 bisa bentrok dengan DLL torch cu130); dibuang `snorkel` (rilis Feb 2024, mungkin tidak jalan), `fasttext-wheel` (bentrok dengan `fast-langdetect`, bug `predict` di NumPy 2), `flashtext` dan `compress-fasttext` (tidak ada wheel Windows), `peft`, `optimum`, `catboost`.

### 6.5 Status pengujian

| Komponen | Status |
|---|---|
| `--check` semua 53 item (link, ID model, ukuran, torch di indeks PyTorch) | **Lolos, 0 gagal** [Ukur] |
| Bug `optimizer.pt` 890 MB di repo IndoBERTweet | Ditemukan dan diperbaiki; sekarang 445 MB |
| Unduhan nyata: e5-small (31 file), IndoBERTweet, F4, lexicon, stopword (ekstrak zip), MBG TikTok, MBG X, sha256 | **Lolos** [Ukur] |
| Installer pada bundle mini (tolak bundle korup, venv, `--no-index`, salin data, `env.ps1`, pemilih varian torch) | **Lolos** [Ukur] |
| Dry-run resolve 195 paket + `torch 2.14.1+cu130` | **Lolos tanpa konflik, semua wheel** [Ukur] |
| Kode benchmark `03_verify_offline.py` | Jalan di CPU (angkanya tidak berarti) |
| Build wheelhouse penuh (~8 GB) | **[Belum teruji]** |
| Pemasangan torch CUDA dan GPU terbaca | **[Belum teruji]** |
| Benchmark di RTX 3070 | **[Belum teruji]** |
| BERTopic/UMAP/HDBSCAN pada 500.000 baris | **[Belum teruji]**; benchmark RAM ditulis tapi tidak pernah jalan |
| Berkas `onnx/model_quint8_avx2.onnx` di item F5 | Belum diverifikasi, cadangan CPU saja |

**Versi kunci hasil resolve:** torch 2.14.1+cu130, numpy 2.5.4, scikit-learn 1.9.1, transformers 5.22.1, sentence-transformers 6.2.0, bertopic 0.17.4, hdbscan 0.8.44, umap-learn 0.5.12, xgboost 3.4.1, lightgbm 4.7.0, kaleido 1.5.0, onnxruntime 1.31.1.

### 6.6 Jebakan teknis

- PyPI untuk Windows memberi wheel torch **CPU** (124 MB). Itulah sebabnya laptop Ivan CPU-only. Selalu paksa `+cu130` lewat constraint dan indeks PyTorch.
- Wheelhouse hanya untuk Python **3.12 Windows x64**.
- `kaleido` butuh Chrome terpasang; kalau tidak ada, pakai matplotlib untuk gambar.
- `AutoModel` membuang head classifier; model sentimen harus dimuat dengan `AutoModelForSequenceClassification`.
- Muat model lewat path lokal dengan `local_files_only=True`; jangan pakai `trust_remote_code=True`; pakai `sklearn.metrics` alih-alih `evaluate.load`.
- Skrip dataset Hugging Face (`load_dataset` dengan loading script) gagal di `datasets` 5.1.0; pakai file GitHub raw atau parquet.
- Variabel lingkungan Hugging Face dibaca saat import; di notebook, set di sel pertama atau jalankan lewat `start_jupyter.bat`.
- RAPIDS/cuML tidak tersedia di Windows 10 (hanya WSL2/Win11).
- Satu `torch-*.whl` saja boleh ada di wheelhouse utama (script menolak kalau lebih).
- Prefiks `query:`/`passage:` untuk E5 berasal dari ingatan; cek di kartu model.

---

## 7. Strategi 5 jam

Timeline ini **[Penilaian]**, dengan T0 = 13.00 WIB tanggal 11 November. Sesuaikan setelah latihan.

| T (menit) | Kerja | Gate |
|---|---|---|
| 0–10 | Baca soal dua kali, kartu masalah, tanya aturan, jalankan cek environment, salin data mentah read-only | Metrik, format, tenggat, batas submission tertulis |
| 10–25 | Muat ke parquet, EDA cepat, replika metrik | Sensus label, histogram panjang token |
| 25–40 | Dedup, grup, cek cara test diambil, fold/holdout | **G1 T40: pilih cabang task** |
| 40–70 | TF-IDF + linear, validator, **submission pertama** | **G2 T70: selisih lokal vs platform masuk akal? kalau tidak, perbaiki dulu** |
| 70–90 | Noise floor, decision layer, kurva belajar, baca 50 error acak | Batas penerimaan, file cadangan |
| 90–105 | Uji throughput BERT, pilot | **G3 T105: lanjut/kecilkan/lewati BERT** |
| 105–195 | Satu run GPU berbatas waktu dengan checkpoint; CPU untuk audit error, grafik, draf slide | **G4 T150: proyeksi tepat waktu dan validasi naik?** |
| 195–225 | Inferensi test, blend pada holdout | **G5 T225** |
| 225–240 | Decision layer bersarang, kandidat A (berprinsip) dan B (lindung nilai) | File tervalidasi + hash |
| **240** | **Freeze model** | |
| 240–280 | Reproduksi sampel kecil, catatan metode, **selesaikan slide** | |
| 280–290 | Unggah final, cek penerimaan, tangkapan layar | Jangan tunggu menit terakhir |
| 290–300 | Cadangan, tidak mengubah kode | |

**Kill rule BERT:** kalau pada 25% anggaran waktu macro-F1 validasi di bawah TF-IDF pada baris yang sama dan tidak naik, kecilkan atau hentikan. Ini gerbang sumber daya, bukan vonis ilmiah.

**Resep fine-tune default:** `max_len` dari histogram 20.000 sampel (cakup ~95%, kemungkinan 64–128; median komentar TikTok MBG 42 karakter); dynamic padding dan length bucketing; gradient accumulation sampai efektif 16–32; LR 2e-5; warmup 5–10%; layerwise LR decay 0,95; validasi tiap 0,1–0,25 epoch pada 5.000–10.000 baris tetap; 1 epoch untuk ≥ 50.000 baris; `num_workers=0` di Windows; `save_total_limit=2`; satu resep saja. Dengan bf16 di RTX 3070 batch bisa jauh lebih besar dari angka laptop.

**Validasi dan keputusan:**
- Duplikat bisa mengubah kesimpulan: pada 19 dataset sosial, peringkat model berubah di 17 setelah penanganan duplikat. Di set TikTok MBG publik, 40,9% baris mengulang `comment_id` dan satu video memuat 71% komentar unik [Ukur].
- Laporkan skor random dan grup sejak baseline pertama; cek binomial pada pasangan yang muncul tepat dua kali memutuskan mana yang dipakai. Group CV bisa salah kalau test memang acak.
- Macro-F1 multi-kelas: pengali per kelas dengan coordinate ascent, bersarang, setelah blend. Biner: threshold dari kurva OOF yang dihaluskan.
- Blend multi-kelas dengan rata-rata probabilitas, biner dengan rata-rata rank; terima hanya kalau selisih > 2 SE noise floor. Jangan refit setelah threshold ditetapkan; jangan stacking kecuali log-loss.
- Mulai tanpa class weight + decision layer (pada IF26, `[1,1]` + threshold mengungguli `[4,1]`).

**Baseline TF-IDF [Ukur di laptop Ivan, bukan hasil publikasi, tanpa transformer dan tanpa data MBG]:**
- **Jangan buang stopword**: macro-F1 turun di semua 10 fold pada tiga dataset (−0,0493 SmSA, −0,0176 EmoT, −0,0124 Ibrohim). Daftar stopword memuat `tidak`, `bukan`, `jangan`, `belum`, `kurang`. Pakai hanya untuk kata topik dan word cloud.
- Word + char n-gram tanpa preprocessing naik +0,0173 (EmoT) dan +0,0100 (Ibrohim).
- Normalisasi slang berkisar −0,0033 sampai +0,0010; stemming sekitar netral. Jadikan saklar, satu uji saja.
- Stemming Sastrawi: lakukan pada kosakata unik lalu petakan (36.346 kata ≈ 4,2 detik; pemetaan 500.000 dokumen ≈ 2 detik).
- Kamus-alay mengubah 13,1% token di EmoT dan 9,7% di Ibrohim, tetapi hanya 0,6% di SmSA. Istilah MBG (SPPG, BGN) kemungkinan tidak ada di kamus lama.
- **InSet bukan pemberi label**: macro-F1 hanya 0,24–0,35 di SmSA, melabeli 76–91% baris negatif (`yang`, `tidak`, `sangat` berbobot −5). Sebagai fitur efeknya kecil (+0,0053 di SmSA).

**Memori dan skala:** `HashingVectorizer(n_features=2**20, alternate_sign=False)` + `SGDClassifier(loss="log_loss")` dengan `partial_fit` untuk 500k baris; baru coba TF-IDF + `LinearSVC` dengan `n_jobs` terbatas (Windows menyalin data per proses). Embedding 384 dimensi × 500.000 = 768 MB, tulis ke memmap. Kode harus di dalam `if __name__ == '__main__':` karena Windows memakai `spawn`. Dengan RAM 32 GB di PC panitia batasnya jauh lebih longgar, tetapi tetap ukur.

**BERTopic pada 500.000 baris belum terbukti.** Urutan yang disarankan: dedup, embedding sekali lalu simpan, UMAP + HDBSCAN hanya pada sampel 20.000–50.000, assign sisanya; `low_memory=True`, `calculate_probabilities=False`, `min_df` tinggi; fallback IncrementalPCA + MiniBatchKMeans. Hindari OCTIS (mematok scikit-learn 1.1.0). Laporkan topik sebagai tema, bukan klaster tegas (silhouette Compfest25 ≈ −0,005).

**Kit kode yang sebaiknya ditulis dan dilatih (belum dibuat):** `00_env_check`, `config`/`common`, `01_load_parquet`, `02_fast_eda`, `03_dedup_groups`, `04_metric`, `05_tfidf_linear`, `06_finetune` (resume), `07_embed_memmap`, `08_head_on_embed`, `09_weak_labels`, `10_setfit_zeroshot`, `11_topic`, `12_decision_layer`, `13_blend`, `14_error_audit`, `15_validate_submission` (kolom, jumlah baris, urutan id, NaN, SHA-256), `16_charts`, `slides_template.pptx`, `RUNBOOK.md` (cetak). Saat ini hanya tiga script environment di atas yang sudah ada.

**Jadwal latihan (D-1 = 10 Nov, Set-Up Work Station):**

| Tanggal | Latihan | Lulus kalau |
|---|---|---|
| 6 Nov | `--check`, unduh profil `plus`, kirim pertanyaan ke panitia | Bundle lengkap |
| 7 Nov | Latihan install offline di laptop, Wi-Fi mati, folder baru, **reboot**, verifikasi lagi | Lolos setelah reboot |
| 8 Nov | Uji skala 500.000 baris sintetis (hanya untuk waktu); latihan 5 jam cabang berlabel (SmSA/EmoT) | Tabel waktu terisi, tanpa OOM, baseline sebelum T70 |
| 9 Nov | Latihan label sedikit, tanpa label (topik, garis waktu, 10 slide, **uji RAM BERTopic**), task kejutan (QA/pasangan/regresi), bunuh dan lanjutkan training | Resume berhasil |
| 10 Nov | Salin ke SSD eksternal; di PC panitia saat Set-Up: pasang, verifikasi, catat benchmark | Semua model termuat dengan `local_files_only=True` |

**Kerja tim:** dua orang → Modeller (GPU, validasi, decision layer) dan Analyst (EDA, gold, audit error, grafik, slide, log submission). Tiga orang → tambah Data/QA. Sinkron tiap 30 menit; orang kedua memeriksa sebelum unggah final. Ukuran tim Ivan belum diketahui.

**Slide (8–10 slide, [Penilaian]):** jawaban satu kalimat; data dan metode termasuk dedup; garis waktu dengan kejadian; peta tema; heatmap aspek × sentimen; dua pendalaman dengan kutipan; validasi gold dan stabilitas; tiga rekomendasi dengan pemilik dan metrik; batasan. Pembeda dari studi MBG yang sudah umum (sentimen + topik): pemisahan volume asli vs copypasta, skor tervalidasi gold, pemisahan "menolak program" dari "minta diperbaiki". Karena data tidak boleh keluar dari PC, siapkan template slide sebelumnya dan masukkan grafik **selagi lomba berjalan**.

---

## 8. Domain MBG

### 8.1 Empat rezim wacana

Model pada satu jendela waktu akan drift: (1) peluncuran 6 Jan 2025 (190 SPPG, 570.000 penerima); (2) puncak keracunan Sep–Okt 2025; (3) guncangan korupsi dan kepemimpinan Jun–Jul 2026; (4) gelombang kedua dan debat anggaran Agu–Okt 2026.

### 8.2 Peristiwa untuk fitur lonjakan

| Tanggal | Peristiwa |
|---|---|
| 17–22 Sep 2025 | Garut 569; Banggai Kepulauan 339; Bandung Barat 1.333 (hitungan lain berbeda) |
| 24 Sep 2025 | Bandung Barat menetapkan KLB; BGN 70 kasus, 5.914 penerima (Jan–Sep) |
| Okt 2025 | Puncak 85 kasus, lalu 40, 12, dan 10 pada Jan 2026 |
| 2–3 Jun 2026 | Dadan Hindayana dan dua deputi dicopot; Kejagung menggeledah; tiga orang jadi tersangka. **Masih dugaan** |
| 22 Jul 2026 | Sudaryono dilantik sebagai Kepala BGN (Kompas menulis menggantikan Dadan langsung; sumber lain menyebut Nanik S. Deyang memimpin sementara) |
| Jul–Agu 2026 | Anggaran 2027: Rp174T (Banggar), Rp270,2T indikatif (BGN), Rp240T sementara |
| 3 Sep 2026 | 5.482 korban, 71 insiden sejak 22 Jul; lebih dari 80% pelanggaran SOP |
| 29 Okt 2026 | Laporan 100 hari Sudaryono, "yayasan calo", "MBG 2.0" (angka antar artikel tidak cocok) |

Skala per 29 Okt 2026: sekitar 26.723 SPPG dan 58 juta penerima setelah ~6 juta duplikat dikoreksi.

### 8.3 Angka yang diperselisihkan, jangan digabung

Korban keracunan: BGN 11.640 (Nov 2025), Kemenkes 50.059 dalam 560 kejadian (2 Sep 2026), JPPI 43.838, KPAI 12.658 anak (2025). Tampilkan per lembaga dengan tanggal. Survei kepuasan juga bukan satu tren (pertanyaan dan sampel berbeda). Proporsi kelas tidak bisa ditebak: Drone Emprit melaporkan 50% positif sepanjang setahun, INDEF 85–97% negatif pada September 2025. **Leksikon yang menyebut Dadan Hindayana sebagai kepala BGN sudah usang.** Jangan menyebut tersangka bersalah; jangan menyebut akun sebagai bot tanpa bukti (pada 12.000 komentar YouTube ditemukan 0 kandidat koordinasi ketat).

### 8.4 Jebakan data yang perlu diduga [Inferensi]

Duplikat antar train/test, template copypasta, baris hanya emoji dan `[Sticker]`, singkatan "MBG" bermakna lain atau spam judi, campuran Jawa dan Sunda, sarkasme, noise label. Deteksi bahasa gagal untuk Jawa/Sunda pada uji lokal 6 kalimat; daftar kata penanda lebih masuk akal.

### 8.5 Leksikon awal (hipotesis untuk label lemah, bukan kebenaran)

Tag: **[D]** terlihat di CSV TikTok publik; **[P]** dari paper/berita; **[I]** saran sendiri belum diverifikasi. Setiap aturan harus diaudit dengan jumlah hit dan presisi pada 30 sampel acak (di MT26, kata `github` dan `nlp` punya 0 hit dan `git` hanya cocok dengan "gitu"). Nilai pada gold yang tidak pernah dipakai menulis aturan.

| Aspek | Kata kunci |
|---|---|
| keamanan_pangan | [D] keracunan, racun, basi, bau, korban, layak, tidak layak; [P] klb, puskesmas, sop, e coli, salmonella; [I] ulat, belatung, berjamur; mual, muntah, diare, pusing |
| kualitas_menu | [D] menu, menunya, burger, ayam, nasi, sayur, susu, kotak, roti, telur/telor, tempe, tahu, ikan, kacang, mie, gula, buah, pisang, jeruk, enak, rasa; [P] ompreng; [I] ultra processed/upf, porsi |
| gizi_kesehatan | [D] gizi, gizinya, bergizi, ahli gizi, dokter, sehat; [I] stunting, protein, kalori, akg, obesitas |
| anggaran | [D] anggaran, dana, uang, buang, pendidikan, mending; [P] apbn, rapbn, pajak, triliun, pagu, efisiensi, pemborosan, refocusing |
| korupsi_tata_kelola | [D] korupsi, ladang, lahan, ajang, berjamaah, proyek, pejabat, untung; [P] koruptor, nepotisme, kolusi, markup, tersangka, kejagung, jual beli titik, yayasan, calo, transparan |
| implementasi_dapur | [D] sppg, dapur, ditutup/tutup; [P] kepala sppg, mitra, distribusi, kapasitas, suspend, investigasi, sanitasi; [I] terlambat, tepat waktu, slhs, relawan |
| pendidikan_guru | [D] guru, gaji, honorer, murid, siswa, sekolah, kantin; [P] santri; [I] pip |
| ekonomi_umkm | [D] umkm, lapangan kerja, pekerja, pedagang, jualan, kantin, catering, harga, naik; [I] petani, peternak, susu lokal |
| politik | [D] prabowo, wowo, dpr, tni, polri, polisi, purnawirawan; [P] gibran, janji kampanye, militer, purbaya, zulhas, sudaryono, nanik, dadan, ombudsman, jppi, kpai; [I] icw, cisdi |
| penolakan | [D] stop, tutup, bubar, evaluasi, ganti, mending; [P] bubarkan; setop, hentikan |
| dukungan | [D] bagus, setuju, program bagus; [P] semoga; [I] lanjut, terima kasih/makasih, alhamdulillah |
| media/hoaks/buzzer | [D] hoax, buzzer, influencer, narasi, digoreng, pendukung, fanatik; [P] hoaks, komdigi, klarifikasi; [I] fitnah, bot |
| Normalisasi ejaan [D] | yg, gk, gw, bgt, tp, klo, udh, dpt, jg, jd, bnyk, karna, krn, utk, tdk, jgn, org, dr, anak2, lgi, bnr |
| Jawa/Sunda | [D] ndk, madak, joss; [I] ora, ndak, mboten, wes, sing, teu, henteu, atuh, euy |

### 8.6 Literatur dan data MBG

Studi terbit dominan pada sentimen 3 kelas dan BERTopic. Contoh skor: IndoBERT akurasi 0,8910 / macro-F1 0,8703 (UPN Jatim); akurasi 70% / macro-F1 0,7002 pada 5.509 komentar YouTube (JSAI). Dataset publik MBG total sekitar 100.000 baris dan sebagian besar berlabel otomatis (bukan gold): TikTok 57.362 baris (`MBG_PROCESSED.csv`, repo `fidopecado/Indonesian-Tiktok-Comments-Sentiment-Analysis`), TikTok Kaggle (`sinryurifal/...`, CC BY-NC-SA 4.0, butuh login), YouTube (briannaufal211; Kaggle dzikriraihan), Instagram 14.051 komentar (Mendeley gfsn8x62dk), X berlabel manual ±1,1k baris (`tasyapndya/free_nutritious_meals_sentiment_x_id_2025`). **Pakai hanya kalau panitia mengizinkan data eksternal.** Taksonomi 19 topik JSAI bisa dipakai untuk penamaan topik.

---

## 9. Pertanyaan ke panitia (sebelum atau saat 10 November)

1. **Boleh mencolok SSD eksternal/USB ke PC saat Set-Up?** Kalau tidak, bagaimana cara memasukkan environment? (menentukan seluruh jalur)
2. Hak admin ada? Python, VC++ Redistributable, driver NVIDIA sudah terpasang? Versi driver (`nvidia-smi`)?
3. Environment yang dipasang saat Set-Up dijamin tetap ada di PC yang sama pada 11 November?
4. Model pre-trained (IndoBERT, sentence-transformers), data eksternal (leksikon, stopword, dataset), dan LLM lokal boleh? Harus dideklarasikan?
5. Dataset ada di disk lokal PC atau di server panitia? Format submission, batas jumlah submission, pembagian public/private, submission mana yang dihitung? Bagaimana cara submit tanpa internet?
6. Boleh memakai test untuk fit TF-IDF, self-training, dedup? Boleh hand-label, termasuk baris test?
7. Kode harus bisa dijalankan ulang? Batas runtime?
8. Slide: bagaimana memindahkan grafik dari PC ke laptop (data tidak boleh keluar)? **Tenggat 21.00 WIB tanggal berapa?** Rubrik dan durasi presentasi? Boleh diedit semalam sebelum 12 Nov?
9. Stop kontak, ekstensi, ponsel, catatan cetak, jumlah laptop per tim?
10. Skema data, bahasa, sumber, jenis label, cara label dibuat?

---

## 10. Jangan

| Jangan | Alasan |
|---|---|
| Berasumsi pretrained/data eksternal/LLM diizinkan | Panduan tidak mengatur; IF26 kehilangan 5 submission karena larangan pretrained |
| Fine-tune 500.000 baris, 5-fold BERT, atau zero-shot semua baris | Waktu; LLM ≈ 139 jam pada 1 detik per baris |
| Menjalankan LLM dan training GPU bersamaan | Risiko OOM |
| Memakai model hoaks siap pakai sebagai pemeriksa fakta | ROC-AUC 0,50 pada benchmark independen |
| Menghapus stopword atau stem sebelum classifier | Uji lokal negatif |
| Menghapus label mencurigakan otomatis | Pakai cleanlab untuk prioritas baca; SD26 membuang 186 gambar yang justru benar |
| Melabeli baris test atau menyetel threshold darinya | Risiko kepatuhan |
| Mengejar gerakan kecil di leaderboard | Noise (CDW2 public LB ±0,009) |
| Mengedit submission di Excel | Merusak id dan encoding |
| Mengunduh repo Hugging Face utuh | 2–10× lipat ukuran (MiniLM-L12: 4,6 GB vs 471 MB) |
| Memasang atau mengubah driver, atau paket baru, di hari terakhir | Crash lebih mahal dari panas |
| Mengandalkan VS Code atau LLM sebagai titik gagal tunggal | JupyterLab dan docs lokal harus cukup |
| Memakai data MBG/eksternal tanpa izin | Aturan belum jelas |

---

## 11. Pelajaran dari portofolio Ivan (relevan untuk strategi)

- Bangun **decision layer** sejak awal dan bersarang (cut point, pengali kelas, quota): sering lebih besar efeknya daripada ganti model.
- **Ukur noise floor** sebelum menilai perbaikan; banyak "peningkatan" terlambat ada di dalam noise.
- Pemulihan struktur tersembunyi (grup, near-duplicate) menentukan CV yang jujur.
- Simpan OOF, prediksi test, dan id di setiap run.
- Satu perubahan per run dengan toggle.
- Prefer rata-rata tanpa parameter (equal/rank) kalau kandidat setara.
- Dry run yang dipangkas pernah membalik kesimpulan empat dari empat kali (CDW1): pilot hanya gerbang sumber daya.
- **Celah portofolio:** belum ada pengukuran bersih fine-tuning transformer Indonesia dan tidak ada proyek NLP mendekati 500k baris. Resep fine-tuning belum terbukti sampai latihan.
- Konflik antar pelajaran: grup near-duplicate di CV (G-08) vs test acak (G-09); keduanya dilaporkan, lalu cek sampling test memutuskan.

---

## 12. Keadaan sistem Ivan dan housekeeping

- Python global di laptop **tidak diubah** selama riset.
- Venv dan unduhan riset ada di folder scratchpad sesi (akan hilang bersama sesi) dan **tidak boleh dijadikan dasar instalasi H-1**; bangun ulang dari wheelhouse. Salah satu venv riset dalam keadaan rusak (instalasi terpotong).
- Cache Hugging Face (`~\.cache\huggingface`) mendapat beberapa entri dataset hasil riset (indonli, indonlu, NusaX-senti, mteb NusaX, fixed_indonlu). Boleh dihapus. Saat latihan, pastikan tidak ada yang diam-diam memakai cache lama karena bisa menyembunyikan file yang hilang.
- File uji dari sesi ini (bundle mini, venv uji, cache pip) sudah dihapus.
- Perangkat lunak yang belum terpasang di laptop: conda, uv, ollama. PyTorch CPU-only.

---

## 13. Daftar tugas berikutnya

1. Kirim pertanyaan ke panitia (bagian 9).
2. Jalankan `01_download_all.py --check`, lalu unduh profil `plus` ke `D:\comp\bundle`.
3. Latihan install offline di laptop (Wi-Fi mati, folder baru, reboot, verifikasi). Perbaiki apa pun yang gagal.
4. Tulis dan latih kit kode (bagian 7) dan template slide.
5. Siapkan SSD eksternal (salin `bundle`), cetak `RUNBOOK`.
6. Saat Set-Up 10 November: pasang, jalankan verifikasi, catat benchmark RTX 3070, lalu perbarui estimasi waktu.

---

## 14. Peta file

```
D:\Lomba\2026\Gemastik\
├─ RINGKASAN_PERSIAPAN_GEMASTIK.md          <- file ini
├─ Buku Saku Gemastik XIX - 2026 portrait.pdf
├─ persiapan_h1\                            <- script download/install/verify
└─ riset\
   ├─ Laporan_lengkap_Persiapan_final_NLP_offline_MBG.md
   └─ catatan_mentah\
      ├─ indonesian_pretrained_models.md      (model Indonesia, embedding, estimasi throughput)
      ├─ local_llm_offline.md                 (LLM lokal, runtime, GGUF)
      ├─ mbg_domain_and_tasks.md              (domain MBG, task, literatur, dataset)
      ├─ indonesian_resources_lexicons.md     (leksikon, preprocessing, uji lokal)
      ├─ offline_environment_packaging.md     (torch/CUDA, wheelhouse, HF offline, Windows)
      └─ strategy_5h_and_past_lessons.md      (strategi 5 jam, pelajaran portofolio)
```

**Catatan tentang laporan lengkap:** isinya ditulis sebelum info PC panitia diketahui. Bagian yang mengasumsikan laptop GTX 1650 (fp32-only, upgrade RAM, page file, estimasi waktu, LLM di 4 GB) sudah digantikan oleh file ini. Daftar paket dan constraint di laporan itu juga sudah diubah di `persiapan_h1`; jika berbeda, `persiapan_h1` yang berlaku.
