// Strategy & Troubleshooting. Angka ambang adalah aturan praktis, bukan hukum.
window.TK = window.TK || {};
TK.strategy = [
  {
    id: "imbalance", title: "Severe class imbalance", icon: "⚖",
    signal: "Kelas terkecil di bawah 5% atau rasio mayor:minor di atas 20:1. Accuracy tinggi, recall kelas minor rendah.",
    first: "Pakai metrik yang benar, split stratified, lalu tuning threshold di OOF. Ini murah dan biasanya memberi gain terbesar.",
    recs: [
      { t: "Threshold tuning di OOF", why: "Model sering sudah mengurutkan dengan baik; yang salah titik potongnya. Tanpa melatih ulang.", cost: "Detik", slot: "imbalance/threshold_tuning" },
      { t: "class_weight='balanced'", why: "Satu parameter pada linear model. Coba setelah threshold; sering tak menambah banyak setelah threshold ditala.", cost: "Rendah", slot: "imbalance/class_weights" },
      { t: "Threshold/multiplier per kelas", why: "Untuk macro-F1 multikelas, argmax mentah jarang optimal.", cost: "Detik", slot: "postprocessing/class_thresholds" },
      { t: "Undersampling kelas mayor", why: "Mempercepat training saat data besar, dan bisa dibagging. Probabilitas perlu dikalibrasi ulang.", cost: "Rendah", slot: "imbalance/resampling" },
      { t: "Focal loss", why: "Untuk transformer saat kelas minor tetap diabaikan walau sudah dibobot.", cost: "Sedang", slot: "imbalance/focal_loss" },
      { t: "SMOTE di fitur padat", why: "Hanya masuk akal di SVD atau embedding. Bukan di TF-IDF mentah: interpolasi sparse menghasilkan dokumen yang tak bermakna.", cost: "Sedang", slot: "imbalance/smote_svd" }
    ],
    avoid: ["Resampling sebelum split: duplikat masuk ke validasi dan skor menjadi palsu.", "Melaporkan accuracy saat metrik sebenarnya macro-F1."]
  },
  {
    id: "shift", title: "Distribution shift", icon: "⇄",
    signal: "AUC adversarial di atas 0.7 (0.5 berarti tidak bisa dibedakan). CV bagus, tetapi skor test jauh lebih buruk.",
    first: "Lihat dulu fitur apa yang membedakan train dan test, lalu buang penyebabnya bila itu artefak (id, tanggal, sumber, format).",
    recs: [
      { t: "Adversarial validation", why: "Satu angka dan daftar fitur yang menjelaskan jenis shift.", cost: "1–2 menit", slot: "distribution_shift/adversarial_validation" },
      { t: "Buang fitur pembeda yang tidak stabil", why: "Kata seperti nama domain atau tanggal bisa dihafal model tanpa bisa digeneralisasi.", cost: "Rendah", slot: "distribution_shift/vocab_shift" },
      { t: "Validasi mirip test", why: "Pilih fold train yang paling mirip test, atau split waktu/sumber bila test berasal dari periode/sumber lain.", cost: "Rendah", slot: "validation/split_time" },
      { t: "Importance weighting", why: "Masuk akal untuk covariate shift ringan sampai sedang; bobot ekstrem meningkatkan varian, jadi clip.", cost: "Rendah", slot: "distribution_shift/importance_weights" },
      { t: "Self-training di test", why: "Pseudo-label membantu saat shift kuat dan penggunaan test tanpa label diizinkan. Cek aturan lomba dulu.", cost: "Sedang", slot: "semi_supervised/pseudo_label" },
      { t: "Fit vocabulary gabungan", why: "Fit TF-IDF di train+test menambah cakupan kata, hanya bila aturan membolehkan pemakaian test.", cost: "Rendah", slot: "distribution_shift/domain_adaptation" }
    ],
    avoid: ["Menyetel apa pun di test berlabel atau public score secara berulang: overfit ke leaderboard.", "Menganggap shift pasti ada karena AUC 0.6; pada dataset besar perbedaan kecil pun terdeteksi. Lihat efeknya ke skor."]
  },
  {
    id: "noisy-labels", title: "Noisy labels", icon: "⚠",
    signal: "Ada teks identik dengan label beda. Kesalahan paling yakin model terlihat benar ketika dibaca manual.",
    first: "Ukur dulu. Baca 30 kesalahan paling yakin di OOF; kalau lebih dari sepertiganya label yang keliru, label cleaning layak dikerjakan.",
    recs: [
      { t: "Conflicting labels", why: "Paling murah dan pasti: satukan teks identik dengan voting mayoritas atau buang.", cost: "Detik", slot: "noisy_data/conflicting_labels" },
      { t: "Temukan noise dari OOF", why: "Prediksi yakin yang berlawanan dengan label memberi kandidat. Wajib OOF, bukan prediksi di data latih.", cost: "Rendah", slot: "noisy_data/label_noise_oof" },
      { t: "Label smoothing", why: "Melunakkan keyakinan pada label yang salah di transformer; parameter `label_smoothing_factor` di TrainingArguments.", cost: "Rendah", slot: "modeling/transformer_trainer" },
      { t: "Ensemble seed/fold", why: "Rata-rata mengurangi hafalan noise tanpa asumsi tambahan.", cost: "Sedang", slot: "postprocessing/prob_average" }
    ],
    avoid: ["Membersihkan label di fold validasi: skor naik, kemampuan tidak. Jangan ubah data yang dipakai mengukur.", "Membuang semua sampel sulit: yang hilang juga sampel informatif."]
  },
  {
    id: "missing", title: "Missing values & teks kosong", icon: "∅",
    signal: "`isna()` tidak nol, atau teks kosong setelah `strip()`. Waspadai string 'nan', 'null', '-', dan 'N/A'.",
    first: "Pisahkan tiga kasus: teks kosong di test (wajib tetap diprediksi), teks kosong di train (biasanya buang), dan kolom pendukung kosong.",
    recs: [
      { t: "Ganti NaN dengan string kosong atau token", why: "TF-IDF error pada NaN; `fillna('')` menyelamatkan pipeline. Token khusus memungkinkan model belajar pola kosong.", cost: "Detik", slot: "eda/quick_profile" },
      { t: "Prediksi default untuk baris test kosong", why: "Jumlah baris submission harus utuh. Pakai kelas mayoritas atau prior.", cost: "Detik", slot: "postprocessing/rule_overrides" },
      { t: "Indikator kosong sebagai fitur", why: "Kosong bisa bermakna (misal ulasan tanpa isi). Gabungkan dengan judul bila ada.", cost: "Rendah", slot: "feature_engineering/meta_features" }
    ],
    avoid: ["`astype(str)` tanpa `fillna`: NaN berubah menjadi kata 'nan' yang masuk vocabulary.", "Menghapus baris test."]
  },
  {
    id: "duplicates", title: "Duplicate data", icon: "⧉",
    signal: "Duplikat persis setelah normalisasi di atas 1%, atau teks train muncul di test. Template panjang menghasilkan near-duplicate.",
    first: "Duplikat lintas fold membuat CV terlalu optimis. Pakai GroupKFold pada kelompok teks yang sama sebelum percaya skor apa pun.",
    recs: [
      { t: "Deteksi duplikat ternormalisasi", why: "Lowercase dan buang tanda baca menemukan duplikat yang tak terlihat oleh `duplicated()` mentah.", cost: "Detik", slot: "noisy_data/duplicates" },
      { t: "Grup near-duplicate untuk CV", why: "Hash teks ternormalisasi, MinHash, atau klaster kosinus tinggi sebagai group.", cost: "Rendah", slot: "validation/near_dup_groups" },
      { t: "Cek tumpang tindih train-test", why: "Kalau test mengandung teks train, lookup langsung dari label train sering valid dan sangat akurat. Pastikan aturannya membolehkan.", cost: "Detik", slot: "validation/leakage_overlap" },
      { t: "Pisahkan template dan slot", why: "Bila banyak pesan berasal dari template, informasi ada di bagian yang berubah.", cost: "Sedang", slot: "eda/class_keywords" }
    ],
    avoid: ["Drop duplikat di test.", "Random split sebelum dedup: leakage klasik yang menaikkan skor CV."]
  },
  {
    id: "length", title: "Teks terlalu pendek atau panjang", icon: "↕",
    signal: "Median di bawah 5 kata (pendek), atau P95 jauh di atas 256 token (panjang).",
    first: "Lihat persentil, bukan rata-rata. Tentukan `max_length` dari P90–P95 agar tidak membuang waktu GPU untuk padding.",
    recs: [
      { t: "Pendek: TF-IDF char n-gram", why: "Tangguh terhadap typo dan variasi ejaan saat kata terlalu sedikit.", cost: "Rendah", slot: "feature_engineering/tfidf_char" },
      { t: "Pendek: meta features dan embedding", why: "Kalimat pendek cocok untuk sentence embedding karena tidak terpotong.", cost: "Sedang", slot: "feature_engineering/sbert_encode" },
      { t: "Panjang: head+tail truncation", why: "Informasi penting sering ada di awal dan akhir, misalnya ulasan dan berita.", cost: "Rendah", slot: "postprocessing/tta" },
      { t: "Panjang: bagi per chunk lalu agregasi", why: "Encode beberapa potongan lalu rata-rata prediksinya. Mahal, pakai bila P95 sangat panjang.", cost: "Tinggi", slot: "feature_engineering/transformer_embed" },
      { t: "Panjang: TF-IDF dulu", why: "TF-IDF tidak memotong apa pun dan tidak sensitif panjang saat dinormalisasi.", cost: "Rendah", slot: "feature_engineering/tfidf_word" }
    ],
    avoid: ["`max_length=512` otomatis: waktu per batch naik banyak tanpa jaminan skor.", "Stopword removal agresif di teks pendek: kata yang tersisa bisa habis."]
  },
  {
    id: "oov", title: "Banyak out-of-vocabulary words", icon: "?",
    signal: "OOV rate test terhadap train di atas 20%, banyak slang, typo, nama produk, atau kode.",
    first: "Beralih ke representasi tingkat karakter dan subword sebelum mencoba membersihkan semuanya secara manual.",
    recs: [
      { t: "TF-IDF char_wb 2–5", why: "Tahan OOV karena memakai potongan kata. Biasanya pengubah termurah.", cost: "Rendah", slot: "feature_engineering/tfidf_char" },
      { t: "FastText di korpus sendiri", why: "Membentuk vektor dari subword, sehingga kata baru tetap punya vektor.", cost: "Sedang", slot: "feature_engineering/fasttext_train" },
      { t: "Normalisasi slang", why: "Bermanfaat bila slang terbatas dan sering. Pakai batas kata agar 'git' tidak mengganti 'gitu'.", cost: "Rendah", slot: "preprocessing/slang_id" },
      { t: "Transformer", why: "Tokenizer subword mengurangi OOV. Pastikan bahasa model sesuai.", cost: "Tinggi", slot: "modeling/transformer_trainer" }
    ],
    avoid: ["Kamus manual panjang yang tak pernah cocok dengan korpus. Hitung coverage-nya dulu.", "Stemming teks penuh slang: hasilnya bisa makin kacau."]
  },
  {
    id: "multilang", title: "Dataset memiliki banyak bahasa", icon: "🌐",
    signal: "Campuran Indonesia-Inggris atau bahasa daerah. Stopword dan stemmer satu bahasa merusak sebagian data.",
    first: "Periksa porsi tiap bahasa dan apakah bahasa berkorelasi dengan target. Jangan langsung menerjemahkan.",
    recs: [
      { t: "TF-IDF char (language-agnostic)", why: "Tanpa stopword atau stemmer bahasa tertentu. Baseline paling aman.", cost: "Rendah", slot: "feature_engineering/tfidf_char" },
      { t: "Model multibahasa", why: "NusaBERT untuk Jawa/Sunda, XLM-RoBERTa bila dibawa. E5-small untuk embedding.", cost: "Tinggi", slot: "modeling/transformer_trainer" },
      { t: "Deteksi bahasa sebagai fitur", why: "Bisa menjadi fitur atau dasar stratifikasi fold. Latih sendiri tanpa dependency.", cost: "Rendah", slot: "modeling/lang_detect" }
    ],
    avoid: ["Stopword satu bahasa untuk data campuran. Deteksi bahasa pada Jawa/Sunda: gagal pada uji lokal, pakai kata penanda.", "Model satu bahasa bila porsi bahasa lain besar, kecuali diuji per bahasa."]
  },
  {
    id: "no-labels", title: "Label tidak tersedia", icon: "◌",
    signal: "Tidak ada kolom target, atau hanya sebagian kecil yang berlabel.",
    first: "Tanyakan output apa yang dinilai: klaster, topik, kata kunci, atau pasangan? Metrik panitia menentukan semua langkah berikutnya.",
    recs: [
      { t: "TF-IDF + SVD + MiniBatchKMeans", why: "Baseline klaster cepat untuk 500 ribu baris tanpa GPU.", cost: "Rendah", slot: "modeling/cluster_kmeans" },
      { t: "Embedding + HDBSCAN", why: "Jumlah klaster tidak perlu ditentukan, dan ada label noise. Reduksi dulu dan pakai subset pada data besar.", cost: "Sedang", slot: "modeling/cluster_hdbscan" },
      { t: "NMF sebagai topik", why: "Sering lebih mudah diinterpretasi daripada LDA pada teks pendek dan jauh lebih cepat.", cost: "Rendah", slot: "modeling/topic_nmf_lda" },
      { t: "Weak supervision", why: "Bila domain jelas, aturan dan kata kunci bisa menjadi label awal.", cost: "Sedang", slot: "semi_supervised/weak_supervision" }
    ],
    avoid: ["Mengejar silhouette saja: bisa tinggi di klaster yang tak berguna. Selalu baca isi klaster.", "Klaster yang didominasi satu artefak seperti template atau bahasa: cek term teratas."]
  },
  {
    id: "train-test-diff", title: "Train-test berbeda format atau sumber", icon: "≠",
    signal: "Rata-rata panjang beda, kolom beda, sumber atau periode lain, atau label test diberikan di format berbeda.",
    first: "Samakan preprocessing dua sisi dan pastikan semua kolom yang dipakai juga tersedia di test.",
    recs: [
      { t: "Uji panjang dan vocabulary", why: "Dua pemeriksaan tercepat, bisa selesai dalam semenit.", cost: "Detik", slot: "distribution_shift/length_shift" },
      { t: "Group/Time split", why: "Bila test dari sumber atau periode baru, validasi harus mensimulasikan hal yang sama.", cost: "Rendah", slot: "validation/split_time" },
      { t: "Hindari fitur sumber-spesifik", why: "Fitur yang tidak ada di test atau bergeser harus dibuang dari awal.", cost: "Rendah", slot: "distribution_shift/feature_shift" }
    ],
    avoid: ["Fitur dari kolom yang hanya ada di train.", "Random K-Fold bila test dari domain baru: CV tidak akan mengukur hal yang benar."]
  },
  {
    id: "fit", title: "Overfitting & underfitting", icon: "📈",
    signal: "Overfit: skor train jauh di atas validasi. Underfit: keduanya rendah dan hampir sama.",
    first: "Bandingkan skor train dengan OOF. Tentukan arah sebelum mengubah apa pun.",
    recs: [
      { t: "Overfit linear: turunkan C, naikkan min_df", why: "Mengurangi fitur hafalan.", cost: "Detik", slot: "modeling/linear_models" },
      { t: "Overfit transformer: early stopping, 2–3 epoch", why: "Fine-tuning biasanya cepat overfit setelah beberapa epoch.", cost: "Rendah", slot: "modeling/transformer_trainer" },
      { t: "Overfit tree: subsample, batasi num_leaves", why: "Tree di sparse berdimensi tinggi mudah menghafal.", cost: "Rendah", slot: "modeling/tree_sparse" },
      { t: "Underfit: tambah n-gram dan char", why: "Representasi yang lebih kaya sebelum model yang lebih kompleks.", cost: "Rendah", slot: "feature_engineering/tfidf_union" },
      { t: "Underfit: model relasi untuk tugas pasangan", why: "Bag-of-words tidak melihat hubungan dua teks. Buat fitur pasangan atau cross-encoder.", cost: "Sedang", slot: "feature_engineering/pair_features" }
    ],
    avoid: ["Menyimpulkan dari satu fold.", "Menambah kompleksitas sebelum mengecek bug: data bocor atau label salah petakan."]
  },
  {
    id: "vram", title: "Limited GPU memory (OOM)", icon: "▣",
    signal: "`CUDA out of memory`, atau penggunaan VRAM mendekati batas pada `nvidia-smi`.",
    first: "Urutan penyelamat: kecilkan batch, kecilkan max_length, aktifkan bf16, lalu ganti ke model lebih kecil.",
    recs: [
      { t: "Dynamic padding", why: "Padding per batch mengurangi memori dan waktu secara nyata.", cost: "Rendah", slot: "modeling/transformer_trainer" },
      { t: "Mixed precision (bf16)", why: "RTX 3070 mendukung bf16 dan fp16. Mengurangi VRAM dan mempercepat.", cost: "Rendah", slot: "optimization/amp_grad_accum" },
      { t: "Gradient accumulation", why: "Batch kecil dengan efek update setara batch besar.", cost: "Rendah", slot: "optimization/amp_grad_accum" },
      { t: "Auto-retry saat OOM", why: "Batch dikecilkan otomatis, tidak perlu mengawasi.", cost: "Rendah", slot: "optimization/oom_fallback" },
      { t: "Model lebih kecil", why: "E5-small dan NLI lite 47 MB jauh lebih ringan dari XLM-R (1,1 GB). IndoBERT lite hemat VRAM, tidak hemat waktu.", cost: "Rendah", slot: "feature_engineering/sbert_encode" }
    ],
    avoid: ["Melanjutkan kernel yang penuh memori: restart lebih cepat daripada membersihkan manual.", "Prediksi dengan gradien aktif: selalu `torch.inference_mode()`."]
  },
  {
    id: "time", title: "Limited competition time", icon: "⏱",
    signal: "Sisa waktu kurang dari dua kali estimasi training, atau belum ada submission setelah satu jam.",
    first: "Selalu punya submission valid dari TF-IDF + linear sebelum menyentuh GPU. Itu fallback kamu.",
    recs: [
      { t: "Ukur throughput", why: "Jalankan 50 batch, lalu proyeksikan waktu per epoch sebelum komit training panjang.", cost: "1 menit", slot: "optimization/throughput_timer" },
      { t: "Subsample stratified", why: "Ambil 10–20% untuk iterasi, lalu latih penuh hanya versi final.", cost: "Detik", slot: "optimization/subsample_stratified" },
      { t: "Embedding + linear", why: "Pretrained embedding sekali encode, simpan, lalu model cepat bisa diulang berkali-kali.", cost: "Sedang", slot: "optimization/encode_cached" },
      { t: "Simpan artefak tiap run", why: "OOF dan prediksi test tersimpan berarti ensemble tetap mungkin walau waktu habis.", cost: "Detik", slot: "optimization/save_artifacts" }
    ],
    avoid: ["Optimasi hyperparameter panjang: biasanya gain terkecil.", "Memulai training terakhir tanpa sisa waktu untuk inference, plus cadangan."]
  },
  {
    id: "no-pretrained", title: "Pretrained model tidak tersedia", icon: "⛔",
    signal: "`OSError: We couldn't connect to huggingface.co`, tidak ada model di cache, atau aturan melarang pretrained.",
    first: "Tidak perlu panik: dalam dataset besar, TF-IDF word+char + linear/LightGBM sering sudah kompetitif. Cek cache dulu, lalu beralih.",
    recs: [
      { t: "Periksa model lokal", why: "Cari di cache dan folder model yang disediakan panitia sebelum menyerah.", cost: "1 menit", slot: "optimization/gpu_check" },
      { t: "TF-IDF word+char + LR/SVM/LightGBM", why: "Tanpa dependency internet. Gabungan word dan char adalah baseline terkuat.", cost: "Rendah", slot: "feature_engineering/tfidf_union" },
      { t: "Word2Vec/FastText korpus sendiri", why: "Embedding tanpa internet, bisa untuk jaringan kecil atau rata-rata dokumen.", cost: "Sedang", slot: "feature_engineering/fasttext_train" },
      { t: "CNN/BiGRU dari nol", why: "Sebagai anggota ensemble yang beragam, terutama pada data di atas 100 ribu baris.", cost: "Tinggi", slot: "modeling/cnn_text" }
    ],
    avoid: ["Menghabiskan waktu mencoba mengunduh.", "NN dari nol sebagai model utama pada data kecil (di bawah 20 ribu teks pendek): di portofolio kamu ini cenderung hanya menghafal (IF26, MT26)."]
  }
];

// Diagnostics: bagian 7 brief.
TK.diagnostics = [
  {
    id: "imb", title: "A. Class Imbalance",
    items: [
      { n: "Class distribution", read: "Rasio mayor:minor. Di atas 20:1 masuk kategori parah.", act: "Pilih macro-F1/PR-AUC dan stratify.", slot: "eda/label_distribution" },
      { n: "Class weights", read: "Bandingkan dengan dan tanpa bobot di fold yang sama.", act: "Pertahankan bila macro-F1 naik melebihi noise.", slot: "imbalance/class_weights" },
      { n: "Threshold tuning", read: "Kurva F1 terhadap threshold di OOF.", act: "Fit di OOF, terapkan satu threshold ke test.", slot: "imbalance/threshold_tuning" },
      { n: "Random undersampling", read: "Waktu training turun; lihat apakah skor bertahan.", act: "Lakukan hanya di fold train.", slot: "imbalance/resampling" },
      { n: "Oversampling", read: "Cek gap train-validasi; duplikat mudah overfit.", act: "Hanya di fold train.", slot: "imbalance/resampling" },
      { n: "SMOTE", read: "Hanya untuk fitur padat seperti SVD atau embedding.", act: "Gunakan imblearn Pipeline agar tetap per fold.", slot: "imbalance/smote_svd" },
      { n: "Stratified splitting", read: "Rasio kelas tiap fold harus sama.", act: "Standar untuk klasifikasi i.i.d.", slot: "validation/kfold_stratified" },
      { n: "Focal loss", read: "Kelas minor tetap rendah meski dibobot.", act: "Mulai gamma 2.0 di transformer.", slot: "imbalance/focal_loss" }
    ]
  },
  {
    id: "shift", title: "B. Distribution Shift",
    items: [
      { n: "Train-test comparison", read: "AUC ≈ 0.5 mirip, > 0.7 layak diselidiki, > 0.9 berbeda kuat.", act: "Periksa fitur pembeda teratas.", slot: "distribution_shift/adversarial_validation" },
      { n: "Vocabulary shift", read: "Kata dengan perbedaan frekuensi terbesar.", act: "Buang kata artefak atau tangani dengan char n-gram.", slot: "distribution_shift/vocab_shift" },
      { n: "Length distribution shift", read: "Statistik KS dan persentil panjang.", act: "Sesuaikan max_length; hindari fitur panjang bila bergeser.", slot: "distribution_shift/length_shift" },
      { n: "Feature distribution shift", read: "PSI > 0.25 berarti shift besar (aturan praktis).", act: "Buang atau transformasikan fiturnya.", slot: "distribution_shift/feature_shift" },
      { n: "Domain shift", read: "Skor jatuh antar sumber pada leave-one-source-out.", act: "Group split per sumber dan model general.", slot: "validation/kfold_group" },
      { n: "Covariate shift", read: "p(x) berubah, p(y|x) dianggap tetap.", act: "Importance weighting ber-clip.", slot: "distribution_shift/importance_weights" },
      { n: "Importance weighting", read: "Lihat ESS (effective sample size); turun drastis berarti terlalu ekstrem.", act: "Clip bobot pada P1–P99.", slot: "distribution_shift/importance_weights" },
      { n: "Domain adaptation", read: "Gain pada target sebanding biaya?", act: "Mulai dari self-training bila diizinkan.", slot: "distribution_shift/domain_adaptation" }
    ]
  },
  {
    id: "noise", title: "C. Noisy Data & Labels",
    items: [
      { n: "Duplicate detection", read: "Persen duplikat persis dan ternormalisasi.", act: "Dedup train, lalu group teks identik.", slot: "noisy_data/duplicates" },
      { n: "Conflicting labels", read: "Berapa grup teks sama dengan label beda.", act: "Voting mayoritas atau buang.", slot: "noisy_data/conflicting_labels" },
      { n: "Label noise identification", read: "Kandidat dari OOF, urutkan dari keyakinan salah tertinggi.", act: "Baca manual; ubah hanya di train.", slot: "noisy_data/label_noise_oof" },
      { n: "Text anomaly detection", read: "Isolation Forest atau jarak kNN tinggi.", act: "Periksa; jangan buang otomatis dari test.", slot: "modeling/anomaly_text" },
      { n: "Outlier detection", read: "Terlalu pendek, panjang, rasio non-huruf tinggi.", act: "Buang di train bila bukan kasus test.", slot: "noisy_data/text_outliers" },
      { n: "Cleaning strategies", read: "Efek tiap langkah terhadap CV yang sama.", act: "Satu perubahan per eksperimen.", slot: "optimization/save_artifacts" }
    ]
  },
  {
    id: "valid", title: "D. Validation Strategies",
    items: [
      { n: "Train-validation split", read: "Data sangat besar atau model mahal.", act: "Risiko: satu split bisa bias; gunakan untuk iterasi cepat.", leak: "Rendah bila dedup dan stratified.", slot: "validation/split_holdout" },
      { n: "K-Fold", read: "Target numerik i.i.d.", act: "Pakai untuk regresi, dengan bin target bila perlu.", leak: "Bocor bila teks duplikat lintas fold.", slot: "validation/kfold_stratified" },
      { n: "Stratified K-Fold", read: "Klasifikasi i.i.d., terutama imbalance.", act: "Default klasifikasi.", leak: "Bocor bila ada grup tersembunyi.", slot: "validation/kfold_stratified" },
      { n: "Group K-Fold", read: "Teks dari penulis, produk, thread, atau dokumen yang sama.", act: "Group harus sama seperti pemisahan train-test.", leak: "Mencegah hafalan entitas.", slot: "validation/kfold_group" },
      { n: "Stratified Group K-Fold", read: "Group ada dan kelas tidak seimbang.", act: "Pakai `StratifiedGroupKFold`.", leak: "Paling aman untuk kasus ini.", slot: "validation/kfold_strat_group" },
      { n: "Time-based split", read: "Ada waktu dan test di masa depan.", act: "Train lampau, valid terbaru; fit TF-IDF hanya pada lampau.", leak: "Random split di sini adalah kebocoran masa depan.", slot: "validation/split_time" },
      { n: "Cross-validation", read: "Fold + OOF untuk seluruh keputusan.", act: "Fit vectorizer, threshold, dan encoder di dalam fold.", leak: "Fit vectorizer di luar fold membocorkan IDF; biasanya kecil tetapi tetap salah.", slot: "validation/kfold_stratified" }
    ]
  },
  {
    id: "other", title: "E. Other Diagnostics",
    items: [
      { n: "Missing text", read: "`isna().mean()`.", act: "fillna('') dan pastikan baris test tetap ada.", slot: "eda/quick_profile" },
      { n: "Empty strings", read: "`str.strip().eq('')` setelah cleaning juga.", act: "Prediksi default.", slot: "eda/quick_profile" },
      { n: "Extremely short/long", read: "Persentil 1, 50, 90, 95, 99.", act: "Tentukan max_length dan truncation.", slot: "eda/length_stats" },
      { n: "Duplicate samples", read: "Persen duplikat dan lintas-split.", act: "Group CV.", slot: "noisy_data/duplicates" },
      { n: "OOV words", read: "Persen token test yang tidak ada di train.", act: "Di atas 20%: char n-gram atau FastText.", slot: "eda/vocab_stats" },
      { n: "Vocabulary size", read: "Kata unik dan kata dengan df=1.", act: "Set min_df dan max_features.", slot: "eda/vocab_stats" },
      { n: "Rare words", read: "Persen kata unik yang muncul sekali.", act: "`min_df=2` atau `3`; char n-gram bila banyak typo.", slot: "eda/vocab_stats" },
      { n: "Class-specific keywords", read: "Kata pembeda per kelas masuk akal?", act: "Kata aneh menandakan artefak atau kebocoran.", slot: "eda/class_keywords" },
      { n: "Train-validation overlap", read: "Harus 0 teks ternormalisasi yang sama, kecuali test juga overlap.", act: "Pakai group.", slot: "validation/leakage_overlap" }
    ]
  }
];
