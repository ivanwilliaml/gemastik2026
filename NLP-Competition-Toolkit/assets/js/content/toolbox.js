// Aturan preprocessing, metrik, dan perbandingan model.
window.TK = window.TK || {};

TK.preprocessingRules = [
  "Lihat 50 teks mentah dulu. Tanda baca, huruf kapital, atau template mungkin tidak seperti yang kamu kira.",
  "Terapkan fungsi yang sama ke train dan test. Fungsi pembersih tanpa state aman; vectorizer tidak boleh fit di validasi.",
  "Uji satu branch per eksperimen dengan fold yang sama. Preprocessing yang lebih rumit tidak otomatis lebih baik.",
  "Simpan kolom teks asli. Jangan timpa, agar bisa dibandingkan saat error analysis."
];
TK.prepChoice = [
  ["TF-IDF untuk klasifikasi", "Tanpa preprocessing atau A", "[Ukur] Buang stopword merugikan. Slang dan stem hanya saklar.", "doc"],
  ["Topic modeling atau klaster", "B", "Kata umum perlu dibuang agar tema muncul.", "unsup semi"],
  ["Word2Vec/FastText", "A atau C", "Stemming jarang perlu.", "doc unsup"],
  ["Transformer, SBERT", "D", "Ikuti aturan pelatihan model. IndoBERTweet: `@USER`, `HTTPURL`, emoji jadi teks.", "tf unsup sim"],
  ["Fitur pasangan", "A", "Pertahankan angka dan entitas untuk perbandingan.", "pair"],
  ["QA dan NER berbasis offset", "Tanpa", "Cleaning mengubah posisi. Normalisasi hanya sebelum offset dihitung.", "qa ner"]
];

TK.metrics = [
  { group: "Classification", for: "cls pair", slot: "evaluation/metrics_classification", items: [
    { n: "Accuracy", f: "(TP+TN) / semua", when: "Kelas seimbang dan biaya salah setara.", trap: "Menyesatkan saat imbalance." },
    { n: "Precision", f: "TP / (TP+FP)", when: "Salah positif mahal.", trap: "Bisa dinaikkan dengan menaikkan threshold." },
    { n: "Recall", f: "TP / (TP+FN)", when: "Kelewat positif mahal.", trap: "Bisa 1.0 dengan memprediksi semua positif." },
    { n: "F1-score", for: "binary pair sentiment", f: "2PR / (P+R)", when: "Biner dengan kelas positif jelas.", trap: "Sangat tergantung threshold." },
    { n: "Macro-F1", f: "Rata-rata F1 per kelas", when: "Semua kelas setara; imbalance.", trap: "Kelas sangat kecil membuat skor bervariasi." },
    { n: "Weighted-F1", f: "Rata-rata F1 berbobot support", when: "Mengikuti proporsi data.", trap: "Bisa menutupi kelas kecil yang buruk." },
    { n: "ROC-AUC", for: "binary pair", f: "P(skor positif > skor negatif)", when: "Kemampuan mengurutkan, tanpa threshold.", trap: "Optimistis pada imbalance berat." },
    { n: "PR-AUC", for: "binary pair", f: "Luas kurva presisi-recall", when: "Positif langka.", trap: "Baseline sama dengan prevalensi, bukan 0.5." },
    { n: "Log Loss", f: "−mean(y log p)", when: "Kualitas probabilitas.", trap: "Prediksi sangat yakin dan salah sangat menghukum; clip." },
    { n: "Confusion Matrix", f: "Tabel kelas asli × prediksi", when: "Mencari pasangan kelas yang tertukar.", trap: "Normalisasi per baris agar kelas kecil terlihat." },
    { n: "Top-k Accuracy", for: "intent multiclass nextword", f: "Jawaban benar ada di k prediksi teratas", when: "Intent banyak kelas, next word.", trap: "Tentukan k dari aturan." }
  ]},
  { group: "Multilabel & ABSA", for: "mlabel absa", slot: "evaluation/metrics_classification", items: [
    { n: "Micro-F1", f: "F1 dari TP, FP, FN semua label", when: "Label umum lebih berpengaruh.", trap: "Label jarang bisa diabaikan tanpa terlihat." },
    { n: "Macro-F1 per label", f: "Rata-rata F1 tiap label", when: "Semua label setara, juga pasangan aspek-sentimen.", trap: "Label jarang membuat skor bervariasi." },
    { n: "Samples-F1", f: "Rata-rata F1 per baris", when: "Kualitas set label tiap dokumen.", trap: "Baris tanpa label perlu `zero_division` jelas." },
    { n: "Hamming Loss", f: "Persen label yang salah", when: "Kesalahan per label dinilai setara.", trap: "Pada label jarang, memprediksi semua nol terlihat bagus." },
    { n: "Subset Accuracy", f: "Seluruh set label tepat", when: "Harus tepat seluruhnya.", trap: "Sangat ketat dan cepat turun saat label banyak." }
  ]},
  { group: "Regression", for: "reg semsim", slot: "evaluation/metrics_regression", items: [
    { n: "QWK", for: "reg", f: "Cohen's kappa dengan bobot kuadrat", when: "Target ordinal, misal skor 1–5.", trap: "Cut point dari OOF sering memberi gain terbesar. Fit nested." },
    { n: "MAE", f: "mean|y−ŷ|", when: "Tahan outlier. Latih dengan objective L1.", trap: "Prediksi optimal adalah median." },
    { n: "MSE", f: "mean(y−ŷ)²", when: "Kesalahan besar sangat dihukum.", trap: "Satuan kuadrat." },
    { n: "RMSE", f: "√MSE", when: "Skala sama dengan target.", trap: "Pakai `root_mean_squared_error`, bukan `squared=False`." },
    { n: "RMSLE", for: "reg", f: "RMSE pada log1p", when: "Target berekor panjang, engagement.", trap: "Target negatif tidak valid." },
    { n: "R²", f: "1 − SSE/SST", when: "Perbandingan dengan prediksi rata-rata.", trap: "Bisa negatif." },
    { n: "MAPE", for: "reg", f: "mean|(y−ŷ)/y|", when: "Target selalu positif dan jauh dari nol.", trap: "Meledak saat y mendekati nol." },
    { n: "Pearson/Spearman", for: "semsim", f: "Korelasi linear/peringkat", when: "Skor kemiripan kontinu.", trap: "Periksa mana yang diminta." }
  ]},
  { group: "Clustering", for: "cluster report", slot: "evaluation/metrics_clustering", items: [
    { n: "Silhouette Score", f: "(b−a)/max(a,b)", when: "Klaster kompak; −1 sampai 1.", trap: "O(n²); gunakan `sample_size`. Bias ke klaster bulat." },
    { n: "Davies-Bouldin", f: "Rasio jarak dalam dan antar klaster", when: "Makin kecil makin baik.", trap: "Bias pada klaster konveks." },
    { n: "Calinski-Harabasz", f: "Rasio dispersi antar dan dalam", when: "Makin besar makin baik.", trap: "Naik mengikuti skala; bandingkan k pada data sama." },
    { n: "Adjusted Rand Index", f: "Kesepakatan pasangan terkoreksi", when: "Ada ground truth, atau uji stabilitas seed.", trap: "Jangan memilih k dari ground truth bila tidak diizinkan." },
    { n: "NMI", f: "Mutual information ternormalisasi", when: "Ada ground truth.", trap: "Cenderung naik saat k lebih banyak." }
  ]},
  { group: "Topic Modeling", for: "topic report", slot: "evaluation/metrics_topic", items: [
    { n: "Topic Coherence", f: "NPMI atau c_v dari kata teratas", when: "Kata dalam topik sering bersama.", trap: "Bisa tinggi di topik tak berguna; baca manual." },
    { n: "Topic Diversity", f: "Persen kata unik di top-N semua topik", when: "Topik tidak saling meniru.", trap: "Tinggi belum tentu koheren." },
    { n: "Topic Interpretability", f: "Penilaian manusia: apakah bisa diberi nama?", when: "Selalu.", trap: "Siapkan contoh dokumen, bukan hanya kata." }
  ]},
  { group: "Similarity / Retrieval", for: "rank keyword", slot: "evaluation/metrics_retrieval", items: [
    { n: "Cosine Similarity", f: "a·b / (‖a‖‖b‖)", when: "Kemiripan arah vektor.", trap: "Normalisasi agar dot product setara." },
    { n: "Precision@K", f: "Relevan di top-K / K", when: "Yang tampil terbatas.", trap: "Maksimumnya bisa di bawah 1." },
    { n: "Recall@K", f: "Relevan di top-K / semua relevan", when: "Tahap kandidat atau first-stage.", trap: "Mudah naik saat K besar." },
    { n: "MAP", f: "Rata-rata average precision", when: "Relevansi biner, banyak jawaban.", trap: "Cek konvensi K dari panitia." },
    { n: "MRR", f: "Rata-rata 1/rank pertama yang relevan", when: "Satu jawaban benar per query.", trap: "Mengabaikan relevan berikutnya." },
    { n: "NDCG@K", f: "DCG@K / IDCG@K", when: "Relevansi bertingkat.", trap: "Periksa rumus gain dan query tanpa relevan." }
  ]},
  { group: "Question Answering", for: "qa", slot: "evaluation/metrics_qa", items: [
    { n: "Exact Match (EM)", f: "Jawaban ternormalisasi sama persis", when: "SQuAD-style.", trap: "Normalisasi Inggris (a, an, the) tidak berlaku. Tulis untuk Indonesia." },
    { n: "F1 token QA", f: "F1 overlap token dengan jawaban emas", when: "Jawaban sebagian dihargai.", trap: "Beberapa jawaban emas: ambil skor maksimal." }
  ]},
  { group: "Text Generation", for: "gen", slot: "evaluation/metrics_rouge", items: [
    { n: "ROUGE", for: "summarization", f: "Overlap n-gram dengan ringkasan rujukan (1, 2, L)", when: "Ringkasan.", trap: "`rouge-score` tidak ada di bundle dan stemmernya untuk Inggris." },
    { n: "Compression Ratio", for: "summarization", f: "Panjang ringkasan / panjang dokumen", when: "Ada batas panjang.", trap: "Skor naik saat ringkasan lebih panjang. Patuhi batas." },
    { n: "Perplexity", for: "nextword", f: "exp(rata-rata negative log-likelihood)", when: "Model bahasa.", trap: "Hanya sebanding pada vocabulary dan tokenisasi yang sama." }
  ]},
  { group: "Terjemahan & Normalisasi", for: "translation normalization", slot: "evaluation/metrics_seq2seq", items: [
    { n: "CER", f: "Jarak edit karakter / panjang rujukan", when: "Normalisasi.", trap: "Makin kecil makin baik." },
    { n: "WER", f: "Jarak edit kata / jumlah kata rujukan", when: "Normalisasi, terjemahan.", trap: "Tokenisasi harus sama." },
    { n: "BLEU", f: "Rata-rata geometrik presisi 1–4 gram × penalti panjang", when: "Terjemahan.", trap: "`sacrebleu` tidak ada di bundle; versi sendiri bisa berbeda." }
  ]},
  { group: "NER", for: "ner", slot: "evaluation/metrics_ner", items: [
    { n: "Entity-level Precision", f: "Entitas tepat / entitas diprediksi", when: "Standar.", trap: "Span dan tipe harus sama persis." },
    { n: "Entity-level Recall", f: "Entitas tepat / entitas asli", when: "Standar.", trap: "Sub-token harus disatukan kembali." },
    { n: "Entity-level F1", f: "2PR / (P+R)", when: "Metrik utama NER.", trap: "Bukan token accuracy." },
    { n: "Exact Match", f: "Span dan tipe sama persis", when: "CoNLL-style.", trap: "Partial match dinilai salah." }
  ]},
  { group: "Anomali", for: "anomaly", slot: "modeling/anomaly_text", items: [
    { n: "Precision@K anomali", f: "Anomali sebenarnya di K teratas", when: "Hanya sebagian yang bisa diperiksa.", trap: "K sesuai kapasitas pemeriksaan." },
    { n: "Injeksi sintetis", f: "Recall pada anomali buatan", when: "Tanpa label.", trap: "Pisahkan seed tuning dan evaluasi." }
  ]}
];

TK.bertVsTfidf = [
  ["NusaX Indonesia, macro-F1", "SVM 78,7", "IndoBERT base 90,0", "≈ +11"],
  ["SmSA, macro-F1", "fastText 82,13", "Base 87,73", "+5,6 (pembanding bukan TF-IDF)"],
  ["Tweet MBG 54.105 baris, akurasi", "SVM 83,39", "IndoBERT 87,60", "+4,2; sumber label tidak disebut"],
  ["TikTok MBG, akurasi", "SVM 0,73", "IndoBERT 0,78", "+5; hanya dari cuplikan"],
  ["Ulasan produk, macro-F1", "LinearSVC 0,5510", "IndoBERT 0,5088", "−4,2; hanya dari cuplikan"]
];

// Paket yang sengaja tidak ada di bundle dan penggantinya.
TK.stdlibOrExternal = { "llama-server": true };
TK.notInBundle = [
  ["catboost", "LightGBM atau XGBoost", "Dibuang dari daftar bundle."],
  ["seqeval", "Fungsi entity F1 sendiri, slot `evaluation/metrics_ner`", "Mudah ditulis dalam beberapa baris."],
  ["rouge-score", "Fungsi ROUGE sendiri, slot `evaluation/metrics_rouge`", "Stemmernya untuk Inggris."],
  ["rank-bm25", "BM25 sparse sendiri, slot `modeling/retrieval_bm25`", "CountVectorizer + rumus BM25 sudah cukup cepat."],
  ["iterative-stratification", "Fungsi iterative stratification sendiri, slot `validation/multilabel_split`", "Tidak perlu paket."],
  ["snorkel", "Labeling function + voting tertimbang", "Rilis Feb 2024, mungkin tidak jalan."],
  ["fasttext", "`gensim.models.FastText`", "Bentrok dengan `fast-langdetect`, bug `predict` di NumPy 2."],
  ["flashtext", "`pyahocorasick`", "Tidak ada wheel Windows."],
  ["cuml (RAPIDS)", "scikit-learn", "Tidak tersedia di Windows 10."],
  ["OCTIS", "`gensim.models.CoherenceModel`", "Mematok scikit-learn 1.1.0."]
];
