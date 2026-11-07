// Decision guide, rencana waktu, hardware, kesiapan offline, jebakan versi.
window.TK = window.TK || {};
TK.wib = function (m) { const t = 13 * 60 + m; return String(Math.floor(t / 60)).padStart(2, "0") + "." + String(t % 60).padStart(2, "0"); };

TK.firstChecklist = [
  { t: "Baca soal dua kali: metrik, format, tenggat, batas submission, aturan pretrained dan pemakaian test.", m: 5 },
  { t: "Cek environment: `nvidia-smi`, versi library, model lokal. Salin data mentah sebagai read-only.", m: 5, slot: "optimization/gpu_check" },
  { t: "Muat ke parquet, lihat shape, kolom, tipe, baca 50 teks mentah.", m: 5, slot: "data_loading/load_table" },
  { t: "Tulis metrik panitia sebagai fungsi, uji dengan contoh kecil.", m: 5, slot: "evaluation/metrics_classification" },
  { t: "EDA cepat: sensus label dan histogram panjang token. Pilih task di bilah atas.", m: 5, slot: "eda/quick_profile" }
];

TK.timePlan = [
  { from: 0, to: 10, name: "Baca soal, cek environment", out: "Kartu masalah, tanya aturan, salin data mentah read-only.", stop: "Metrik, format, tenggat, dan batas submission tertulis." },
  { from: 10, to: 25, name: "Muat data, EDA, replika metrik", out: "Parquet, sensus label, histogram panjang token.", stop: "Tidak melatih model besar." },
  { from: 25, to: 40, name: "Dedup, grup, fold", out: "Cek cara test diambil, fold atau holdout.", stop: "**G1 T40:** pilih cabang task." },
  { from: 40, to: 70, name: "TF-IDF + linear, submission pertama", out: "Validator dan submission tervalidasi.", stop: "**G2 T70:** selisih lokal vs platform masuk akal? Kalau tidak, perbaiki dulu." },
  { from: 70, to: 90, name: "Noise floor, decision layer", out: "Kurva belajar, baca 50 error acak, batas penerimaan.", stop: "File cadangan tersimpan." },
  { from: 90, to: 105, name: "Uji throughput BERT, pilot", out: "Proyeksi waktu dari 50 batch.", stop: "**G3 T105:** lanjut, kecilkan, atau lewati BERT." },
  { from: 105, to: 195, name: "Satu run GPU berbatas waktu", out: "Checkpoint aktif. CPU untuk audit error, grafik, draf slide.", stop: "**G4 T150:** proyeksi tepat waktu dan validasi naik?" },
  { from: 195, to: 225, name: "Inferensi test, blend", out: "Blend pada holdout.", stop: "**G5 T225.**" },
  { from: 225, to: 240, name: "Decision layer bersarang", out: "Kandidat A (berprinsip) dan B (lindung nilai).", stop: "**T240 freeze model.** File tervalidasi + hash." },
  { from: 240, to: 280, name: "Selesaikan slide", out: "Reproduksi sampel kecil, catatan metode.", stop: "Konten slide harus selesai sebelum kompetisi berakhir." },
  { from: 280, to: 290, name: "Unggah final", out: "Cek penerimaan, tangkapan layar.", stop: "Jangan tunggu menit terakhir." },
  { from: 290, to: 300, name: "Cadangan", out: "Tidak mengubah kode.", stop: "Selesai 18.00 WIB." }
];

// Pohon keputusan: node {q, opts:[{a, to}]} atau daun {task, base, metric}
TK.decisionTree = {
  q: "Apa bentuk target?",
  opts: [
    { a: "Kategorikal, per baris", to: {
      q: "Berapa label per baris?",
      opts: [
        { a: "Satu dari 2 kelas", to: { leaf: true, title: "Binary", task: "binary", base: "TF-IDF word+char + LR", metric: "F1, AUC" } },
        { a: "Satu dari >2 kelas", to: { leaf: true, title: "Multiclass", task: "multiclass", base: "TF-IDF + LinearSVC/LR", metric: "Macro-F1" } },
        { a: "Opini 3 kelas", to: { leaf: true, title: "Sentimen", task: "sentiment", base: "TF-IDF tanpa buang stopword", metric: "Macro-F1" } },
        { a: "Lebih dari satu label", to: { leaf: true, title: "Multilabel", task: "multilabel", base: "OvR + threshold per label", metric: "Micro/Macro-F1" } },
        { a: "Sentimen per aspek", to: { leaf: true, title: "ABSA", task: "absa", base: "Model per aspek", metric: "Macro-F1 pasangan" } }
      ] } },
    { a: "Numerik", to: { leaf: true, title: "Regresi", task: "regression", base: "TF-IDF + Ridge, log1p", metric: "RMSE, MAE" } },
    { a: "Dua kolom teks", to: {
      q: "Apa yang diprediksi?",
      opts: [
        { a: "Kelas hubungan", to: { leaf: true, title: "Pair classification", task: "pair", base: "Fitur pasangan + LightGBM", metric: "Macro-F1" } },
        { a: "Pertanyaan sama?", to: { leaf: true, title: "Duplicate question", task: "dupq", base: "Fitur pasangan + LightGBM", metric: "Log Loss, F1" } },
        { a: "Skor kemiripan", to: { leaf: true, title: "Semantic similarity", task: "semsim", base: "Kosinus + Ridge", metric: "Pearson, Spearman" } },
        { a: "Relevansi per query", to: { leaf: true, title: "Ranking", task: "ranking", base: "BM25, lalu LGBMRanker", metric: "NDCG@K" } },
        { a: "Jawaban dari konteks", to: { leaf: true, title: "Question Answering", task: "qa", base: "Ekstraktif TF-IDF, IndoBERT-QA", metric: "EM, F1 token" } }
      ] } },
    { a: "Tag per token", to: { leaf: true, title: "NER", task: "ner", base: "Most-frequent-tag + aturan", metric: "Entity F1" } },
    { a: "Teks (ringkasan)", to: { leaf: true, title: "Summarization", task: "summarization", base: "Lead-N, TextRank", metric: "ROUGE" } },
    { a: "Sebagian berlabel", to: {
      q: "Punya aturan atau kata kunci domain?",
      opts: [
        { a: "Label sedikit, unlabeled banyak", to: { leaf: true, title: "Pseudo-labeling", task: "pseudo", base: "Baseline → keyakinan ≥ 0.95", metric: "Metrik utama, label asli" } },
        { a: "Tidak ada label, ada aturan", to: { leaf: true, title: "Weak supervision", task: "weak", base: "Labeling function + voting", metric: "Coverage, downstream" } }
      ] } },
    { a: "Tidak ada target", to: {
      q: "Apa output yang diminta?",
      opts: [
        { a: "Hanya laporan", to: { leaf: true, title: "Laporan wawasan", task: "report", base: "Dedup, embedding, topik, garis waktu", metric: "Dinilai juri" } },
        { a: "Kelompok dokumen", to: { leaf: true, title: "Clustering", task: "doc_cluster", base: "TF-IDF + SVD + KMeans", metric: "Silhouette, ARI" } },
        { a: "Tema", to: { leaf: true, title: "Topic modeling", task: "topic", base: "TF-IDF + NMF", metric: "Coherence, Diversity" } },
        { a: "Kata kunci", to: { leaf: true, title: "Keyword", task: "keyword", base: "TF-IDF top n-gram", metric: "P@K, manual" } },
        { a: "Dokumen relevan", to: { leaf: true, title: "Retrieval", task: "ir", base: "BM25", metric: "Recall@K, MRR" } },
        { a: "Teks mencurigakan", to: { leaf: true, title: "Anomali", task: "anomaly", base: "IForest di SVD", metric: "P@K, injeksi" } }
      ] } }
  ]
};

TK.decisionSteps = [
  { n: 1, t: "Inspect Dataset", items: ["Adakah target atau label?", "Apa tipe target?", "Berapa jumlah baris?", "Struktur kolom seperti apa?", "Adakah ID?", "Adakah split train-test? Bagaimana test diambil?"], slot: "eda/quick_profile" },
  { n: 2, t: "Identify Task", items: ["Target kategorikal → Classification", "Target numerik → Regression", "Tanpa target → Unsupervised", "Pasangan teks → Similarity/Pair", "Konteks + pertanyaan → QA", "Sequence dan annotation entitas → NER"], slot: "data_loading/detect_columns" },
  { n: 3, t: "Diagnose Dataset", items: ["Imbalance", "Duplikat dan overlap", "Missing dan kosong", "Distribution shift", "Panjang teks", "Vocabulary dan OOV", "Asal label: uji kata kunci"], slot: "distribution_shift/adversarial_validation" },
  { n: 4, t: "Choose Baseline", items: ["Paling cepat yang valid, plus decision layer", "Simpan OOF dan prediksi test", "Kirim submission pertama sebelum T70"], slot: "optimization/save_artifacts" },
  { n: 5, t: "Improve", items: ["Satu perubahan per eksperimen", "Bandingkan dengan noise floor", "Prioritas: decision layer, validasi, representasi baru, lalu tuning"], slot: "postprocessing/prob_average" }
];

TK.quickRef = [
  ["Target 2 kelas, teks ulasan", "binary", "TF-IDF word+char + LR", "F1 atau AUC", "Threshold OOF"],
  ["Target banyak kelas, berita", "multiclass", "TF-IDF + LinearSVC", "Macro-F1", "Pengali kelas"],
  ["Opini media sosial", "sentiment", "TF-IDF tanpa buang stopword", "Macro-F1", "Emoji tetap ada"],
  ["Pro, kontra, netral terhadap MBG", "stance", "TF-IDF + LR", "Macro-F1", "Sentimen ≠ stance"],
  ["Emosi, toksisitas", "emotion", "TF-IDF + LR, model siap pakai", "Macro-F1", "Cek skema label"],
  ["List label per baris", "multilabel", "OvR + LR", "Micro-F1", "Threshold per label"],
  ["Kolom per aspek", "absa", "Model per aspek", "Macro-F1 pasangan", "Baris tanpa aspek"],
  ["Nilai rating atau skor", "regression", "TF-IDF + Ridge", "RMSE, MAE", "Log1p, clip, cut point"],
  ["SMS/email", "spam", "Char TF-IDF + ComplementNB", "F1 spam", "Dedup"],
  ["Dua kolom teks + label", "pair", "Fitur pasangan + LightGBM", "Macro-F1", "Group per teks"],
  ["Berita valid vs palsu", "fakenews", "TF-IDF judul + isi", "F1", "Periksa kata artefak"],
  ["Ucapan pendek, banyak kelas", "intent", "Char+word + LinearSVC", "Accuracy", "Stopword tetap"],
  ["Query + dokumen", "ranking", "BM25 + LGBMRanker", "NDCG@K", "Fold per query"],
  ["Konteks + pertanyaan + jawaban", "qa", "Ekstraktif, lalu Indobert-QA", "EM, F1 token", "Stride, decode span"],
  ["Tanpa target, cari tema", "topic", "TF-IDF + NMF", "Coherence", "Baca topik"],
  ["Tanpa target, bentuk grup", "doc_cluster", "SVD + MiniBatchKMeans", "Silhouette", "L2 normalize"],
  ["Tanpa label, diminta laporan", "report", "Dedup, topik, garis waktu", "Juri", "Gold 150–300"],
  ["Token + tag BIO", "ner", "Baseline → token classification", "Entity F1", "Alignment"],
  ["Banyak data tanpa label", "pseudo", "Keyakinan ≥ 0.95", "Metrik utama", "Label asli saja"],
  ["Teks panjang + ringkasan", "summarization", "Lead-N, TextRank", "ROUGE", "Cek aturan model"]
];

// Dari skill ml-problem-solving dan ringkasan persiapan milik user.
TK.lessons = [
  { t: "Bangun decision layer lebih awal dan nested.", d: "Threshold dan cut point sering memberi gain lebih besar daripada mengganti model. Di DataQuest, cut point QWK memberi +0.055, 5,6× rentang antar model.", tag: "G-03" },
  { t: "Ukur noise floor dulu.", d: "Banyak perbaikan akhir sebenarnya ada di dalam noise.", tag: "G-15" },
  { t: "Lihat teks mentah sebelum merancang.", d: "IF26 tidak memiliki tanda baca sehingga ide berbasis kalimat gagal. MT26 punya lebih dari 50% pesan duplikat karena template.", tag: "IF26, MT26" },
  { t: "Modelkan hubungan untuk tugas pasangan.", d: "Bag-of-words hampir tidak berguna untuk konsistensi judul-isi (AUC 0.55 di IF26). Fitur relasi yang menentukan.", tag: "IF26" },
  { t: "NN dari nol pada data kecil cenderung menghafal.", d: "Pada ≤ 20 ribu teks pendek, skornya rendah (IF26 0.52, MT26 MLP 0.639). Pakai sebagai anggota diversity pada data besar.", tag: "IF26, MT26" },
  { t: "Diversity representasi, bukan sekadar algoritma.", d: "Nilai anggota dari gain ensemble yang jujur, bukan skor tunggalnya.", tag: "G-34" },
  { t: "Pilih rata-rata tanpa parameter saat kandidat hampir seri.", d: "Bobot dan kalibrator yang difit mudah overfit ke OOF.", tag: "G-35" },
  { t: "Di bawah shift kuat, self-training bisa unggul.", d: "ICAIC +0.013 dan Schematics +0.030, selama aturan lomba membolehkan.", tag: "G-39" },
  { t: "Fitur linguistik yang lebih 'benar' bisa memperburuk.", d: "Pada IF26 terjadi empat kali, karena fitur bekerja berbeda pada kelas mayoritas.", tag: "IF26" },
  { t: "Patuhi aturan pretrained sejak awal.", d: "IFest kehilangan 5 submission karena pelanggaran aturan model.", tag: "IF26" },
  { t: "Dry run bukan penentu keputusan.", d: "Dry run yang dipangkas pernah membalik kesimpulan empat dari empat kali. Pilot hanya gerbang sumber daya.", tag: "CDW1" },
  { t: "Celah portofolio: BERT Indonesia dan skala 500 ribu.", d: "Belum ada pengukuran bersih fine-tuning transformer Indonesia dan belum ada proyek NLP mendekati 500 ribu baris. Resepnya belum terbukti sampai latihan.", tag: "Ringkasan" }
];

TK.hardware = {
  profiles: [
    { n: "RTX 3070 (PC panitia)", rec: ["Umumnya 8 GB VRAM; pastikan dengan `nvidia-smi`", "Ampere mendukung bf16 dan fp16", "Fine-tune sampel, bukan 500 ribu baris", "Encode embedding seluruh data sekali, simpan memmap", "Catat throughput dari `03_verify_offline.py` saat Set-Up"], avoid: "Memakai estimasi waktu GTX 1650. LLM dan training GPU bersamaan." },
    { n: "CPU i7-12700", rec: ["TF-IDF word+char + LR, LinearSVC, atau SGD", "LightGBM pada SVD atau sparse terbatas", "Word2Vec dan FastText korpus sendiri", "Audit error dan grafik selagi GPU training"], avoid: "`n_jobs=-1` pada matriks besar: Windows `spawn` menyalin data." },
    { n: "RAM 32 GB", rec: ["Baca hanya kolom yang dibutuhkan", "`dtype='float32'` pada TF-IDF", "`del` dan `gc.collect()` setelah langkah besar", "Embedding 384 dimensi × 500 ribu = 768 MB, tulis memmap"], avoid: "`.toarray()` pada TF-IDF besar; kosinus n×n." },
    { n: "VRAM terbatas", rec: ["Dynamic padding", "bf16", "Batch kecil + gradient accumulation", "`max_len` dari cakupan 95%", "Satu proses GPU saja"], avoid: "Menyimpan tensor di GPU saat inference. `indobert-lite` tidak menghemat waktu." },
    { n: "Hingga 500 ribu baris", rec: ["Hashing + SGD `partial_fit` sebagai jaring pengaman", "TF-IDF penuh dengan `max_features` terbatas", "Iterasi di subset 10–20%", "Transformer pada 50–100 ribu baris, 1 epoch", "Silhouette dengan `sample_size`"], avoid: "HDBSCAN, UMAP, Agglomerative, t-SNE tanpa sampling. BERTopic penuh [Belum teruji]." },
    { n: "Waktu terbatas", rec: ["Ukur throughput dulu", "Simpan OOF tiap run", "Satu run GPU berbatas waktu", "Submission pertama sebelum T70"], avoid: "Optuna panjang di menit akhir. Paket baru di hari terakhir." }
  ],
  transformer: [
    ["Dynamic padding", "`DataCollatorWithPadding`, tanpa `padding='max_length'`", "Paling murah dan langsung bermanfaat."],
    ["Length bucketing", "`train_sampling_strategy='group_by_length'`", "`group_by_length` sudah tidak ada di versi bundle."],
    ["Mixed precision", "`bf16=True` di `TrainingArguments`", "Didukung Ampere. Matikan bila NaN."],
    ["Gradient accumulation", "`gradient_accumulation_steps` sampai efektif 16–32", "Saat batch kecil karena VRAM."],
    ["Max sequence length", "Cakupan ~95% dari 20.000 sampel, kemungkinan 64–128", "Median komentar TikTok MBG 42 karakter."],
    ["Learning rate", "2e-5, warmup 5–10%, layerwise decay 0,95", "Satu resep saja."],
    ["Epoch", "1 epoch untuk ≥ 50.000 baris", "Validasi tiap 0,1–0,25 epoch pada 5.000–10.000 baris tetap."],
    ["Windows", "`dataloader_num_workers=0`", "Hindari masalah spawn."],
    ["Checkpoint", "`save_total_limit=2`, `resume_from_checkpoint`", "Latih bunuh dan lanjutkan."],
    ["Inferensi", "Sort per panjang, batch, kembalikan urutan", "Hemat padding. Selalu kembalikan urutan asli."]
  ],
  scale: [
    ["≤ 50 ribu", "Seluruh data pada TF-IDF K-Fold; transformer satu fold atau holdout."],
    ["50–200 ribu", "TF-IDF penuh; transformer 1 epoch, max_length moderat."],
    ["200–500 ribu", "TF-IDF atau Hashing penuh; transformer pada 50–100 ribu; blend keduanya."]
  ]
};

TK.readiness = [
  { group: "Sebelum berangkat", items: [
    "Unduh bundle: `python 01_download_all.py --out D:\\comp\\bundle --profile plus`.",
    "Latihan install offline di laptop: Wi-Fi mati, folder baru, reboot, verifikasi.",
    "Latih bunuh dan lanjutkan training (resume).",
    "Salin toolkit ini ke laptop dan SSD. Buka tanpa internet di browser lain.",
    "Cetak `RUNBOOK`: JupyterLab dan docs lokal harus cukup bila VS Code bermasalah."
  ]},
  { group: "Set-Up Work Station (10 Nov)", items: [
    "Tanya: boleh mencolok SSD? Tanpa itu script tidak punya jalur masuk.",
    "`D:\\bundle\\tools\\02_install_offline.bat -Dest C:\\gemastik`, lalu catat benchmark RTX 3070.",
    "Pastikan semua model termuat dengan `local_files_only=True`.",
    "Tanya apakah environment tetap ada pada 11 November.",
    "Catat versi driver dan CUDA. Jangan mengubah driver."
  ]},
  { group: "Hari lomba", items: [
    "Set `HF_HUB_OFFLINE=1` di sel pertama, sebelum import, atau jalankan lewat `start_jupyter.bat`.",
    "Muat dengan path lokal, tanpa `trust_remote_code=True`.",
    "Pakai `sklearn.metrics`, bukan `evaluate.load`.",
    "Kode ber-paralel di dalam `if __name__ == '__main__':`.",
    "Laptop tidak menyentuh dataset. Kode dan toolkit boleh menjadi rujukan bila aturan mengizinkan.",
    "Pasang di path pendek seperti `C:\\gemastik`: path panjang membuat pip gagal menulis file `jupyterlab`."
  ]}
];

// Diverifikasi dengan tools/validate_api.py pada venv hasil pip install requirements.in.
TK.versionTested = "torch 2.14.1+cu130 · numpy 2.5.4 · pandas 3.1.1 · scikit-learn 1.9.1 · transformers 5.22.1 · sentence-transformers 6.2.0 · datasets 5.1.0 · lightgbm 4.7.0 · xgboost 3.4.1 · gensim 4.4.0 · bertopic 0.17.4";
TK.versionTraps = [
  ["scikit-learn", "`LogisticRegression(multi_class=...)`", "Hapus argumen itu.", "Parameter sudah tidak ada."],
  ["scikit-learn", "`mean_squared_error(y, p, squared=False)`", "`root_mean_squared_error(y, p)`", "`squared` sudah hilang."],
  ["scikit-learn", "`OneHotEncoder(sparse=False)`", "`OneHotEncoder(sparse_output=False)`", "Nama parameter berubah."],
  ["scikit-learn", "`CalibratedClassifierCV(base_estimator=m)`", "`CalibratedClassifierCV(estimator=m)`", "Nama parameter berubah."],
  ["scikit-learn", "`CalibratedClassifierCV(m, cv='prefit')`", "`CalibratedClassifierCV(FrozenEstimator(m))`", "Impor dari `sklearn.frozen`."],
  ["scikit-learn", "`SGDClassifier(loss='log')`", "`loss='log_loss'`", "Nama loss berubah."],
  ["transformers", "`evaluation_strategy='epoch'`", "`eval_strategy='epoch'`", "Argumen lama sudah tidak ada."],
  ["transformers", "`Trainer(tokenizer=tok)`", "`Trainer(processing_class=tok)`", "Argumen lama sudah tidak ada."],
  ["transformers", "`overwrite_output_dir=True`", "Hapus argumen itu.", "Sudah tidak ada."],
  ["transformers", "`group_by_length=True`", "`train_sampling_strategy='group_by_length'`", "Berubah di versi 5."],
  ["transformers", "`no_cuda=True`", "`use_cpu=True`", "Sudah tidak ada."],
  ["transformers", "`warmup_ratio=0.1`", "`warmup_steps=0.1`", "`warmup_ratio` sudah tidak ada; `warmup_steps` menerima pecahan."],
  ["transformers", "`logging_dir='logs'`", "Hapus argumen itu.", "Sudah tidak ada."],
  ["datasets", "`load_dataset(x, trust_remote_code=True)`", "Pakai file parquet atau CSV lokal", "Script dataset gagal di 5.1.0."],
  ["lightgbm", "`fit(..., early_stopping_rounds=50, verbose=False)`", "`fit(..., callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])`", "Argumen lama dihapus."],
  ["xgboost", "`fit(..., early_stopping_rounds=50, eval_metric='auc')`", "`XGBClassifier(early_stopping_rounds=50, eval_metric='auc')`", "Argumen `fit` lama tidak ada."],
  ["gensim", "`Word2Vec(size=100, iter=5)`", "`Word2Vec(vector_size=100, epochs=5)`", "Nama parameter berubah."],
  ["pandas", "`df['a'][mask] = 1`", "`df.loc[mask, 'a'] = 1`", "pandas 3 selalu Copy-on-Write: chained assignment tidak mengubah `df`."],
  ["pandas", "`df['text'].dtype == object`", "`pd.api.types.is_string_dtype(df['text'])`", "pandas 3 memakai tipe `str` sebagai default."]
];
TK.versionCheck = "import sys, sklearn, torch, transformers, pandas\nprint(sys.version.split()[0], sklearn.__version__, torch.__version__, transformers.__version__, pandas.__version__)\nprint(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\nprint(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)";
