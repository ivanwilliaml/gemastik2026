# Audit cheatsheet offline

Urutan persiapan berdasarkan karakter data teks dan skenario MBG yang sudah dicatat. Bukan probabilitas/fakta soal panitia.

Ditinjau: 142 file snippet dan 293 saran metode; 43 task dalam 25 alur.
Kode lengkap: 299 bagian dari pipeline/snippet asli. Versi kode pendek tidak digunakan pada tampilan utama.

Audit struktur/kondisi tidak membuktikan skor model pada data baru. Hasil uji perilaku dipisah dalam `toolkit/references/cheatsheet_validation.json`.

## Urutan audit task

| Urutan | Alur | Task | Baris kode inti |
|---|---|---|---|
| 1 | Sentimen / stance / klasifikasi | sentiment, stance, binary, multiclass, spam, emotion, fakenews, intent | 1561 |
| 2 | Multilabel / banyak tag | multilabel | 946 |
| 3 | Sentimen per aspek (ABSA) | absa | 432 |
| 4 | Topik dari teks tanpa label | topic | 501 |
| 5 | Clustering teks | doc_cluster, emb_cluster | 468 |
| 6 | Tren topik / sentimen dan laporan | trend, report | 665 |
| 7 | Tanpa gold / weak supervision | weak | 440 |
| 8 | Gold sedikit / pseudo-label / self-training | pseudo, selftrain, confidence, iterative | 461 |
| 9 | NLI / pair matching / parafrase | pair, consistency, dupq | 644 |
| 10 | Semantic similarity berskor kontinu | semsim | 433 |
| 11 | Retrieval / pencarian dokumen | ir | 382 |
| 12 | Ranking dengan relevance berjenjang | ranking | 382 |
| 13 | Regresi skor / engagement / ordinal | regression | 964 |
| 14 | Representasi / kemiripan dokumen / reduksi | doc_sim, repr, dimred | 382 |
| 15 | Anomali / spam tak berlabel | anomaly | 305 |
| 16 | Kata kunci tanpa label | keyword | 220 |
| 17 | Question answering / span | qa | 213 |
| 18 | NER / term aspek / keyphrase BIO | ner, ate, keyphrase | 240 |
| 19 | POS tagging | pos | 240 |
| 20 | Ringkasan ekstraktif / abstraktif | summarization | 192 |
| 21 | Ekstraksi informasi dan bukti domain | mbg_extract | 795 |
| 22 | Normalisasi sebagai task output | normalization | 257 |
| 23 | Terjemahan | translation | 257 |
| 24 | Prediksi kata berikutnya | nextword | 116 |
| 25 | Identifikasi bahasa | langdetect | 1561 |

## Temuan dan keputusan per file

| Sumber | Baris lama | Kartu pengganti | Keputusan / kondisi |
|---|---|---|---|
| `data_loading/detect_columns` | 21 | snippet:data_loading/detect_columns | Tebakan ID/target dari kardinalitas tidak boleh menentukan schema otomatis; tetapkan kolom dari soal. |
| `data_loading/duckdb_query` | 15 | snippet:data_loading/duckdb_query | SQL opsional file besar, identifier kolom perlu escaping; bukan helper wajib pembacaan kecil. |
| `data_loading/load_conll` | 28 | snippet:data_loading/load_conll | Parser CoNLL tetap referensi format khusus; jangan hilangkan batas dokumen atau mengubah offset. |
| `data_loading/load_table` | 27 | snippet:data_loading/load_table | Cache parquet bisa kedaluwarsa dan on_bad_lines bisa kehilangan baris; pembacaan eksplisit lebih mudah diaudit. |
| `data_loading/reduce_memory` | 16 | snippet:data_loading/reduce_memory | Downcast dapat mengubah presisi; category tidak wajib untuk teks, ukur RAM sebelum menyalin DataFrame. |
| `distribution_shift/adversarial_validation` | 22 | snippet:distribution_shift/adversarial_validation | Vectorizer lama fit gabungan sebelum CV domain; diagnostik transduktif bukan estimasi generalisasi induktif. |
| `distribution_shift/domain_adaptation` | 17 | snippet:distribution_shift/domain_adaptation | Fit gabungan train-test bersifat transduktif dan bersyarat aturan; CORAL dense bukan baseline yang wajib diketik. |
| `distribution_shift/feature_shift` | 28 | snippet:distribution_shift/feature_shift | PSI konstan/kosong dan smoothing centered perlu perhatian; waktu online tidak boleh memakai periode masa depan. |
| `distribution_shift/importance_weights` | 15 | snippet:distribution_shift/importance_weights | Density ratio perlu probabilitas domain out-of-fold terkalibrasi dan ESS; tidak default pada data tak dikenal. |
| `distribution_shift/length_shift` | 11 | snippet:distribution_shift/length_shift | KS p-value dipengaruhi jumlah data; laporkan besar shift/persentil dan jangan menganggap p kecil bukti dampak model. |
| `distribution_shift/vocab_shift` | 15 | snippet:distribution_shift/vocab_shift | Log-ratio vocabulary hanya deskriptif; jangan otomatis menghapus kata khas test atau mengoptimasi dari label test. |
| `domain/extract_structured` | 68 | snippet:domain/extract_structured | Aturan uang/tanggal/NER domain panjang; output perlu bukti teks dan audit format lokal, bukan dianggap fakta. |
| `domain/label_source` | 78 | snippet:domain/label_source | Agreement tinggi tidak membuktikan asal label; klaim lama dilunakkan dan heuristik bukan gold. |
| `domain/language_mix` | 52 | snippet:domain/language_mix | Penanda kata daerah adalah heuristik coverage kecil; teks campuran/pendek perlu audit manusia. |
| `domain/mbg_lexicon` | 94 | snippet:domain/mbg_lexicon | Kamus MBG kondisional, label isu bukan sentimen; abstain saat konflik dan audit precision pada gold independen. |
| `domain/temporal` | 68 | snippet:domain/temporal | Pisahkan volume dan proporsi; topic ID harus beku, dukungan kecil dan perubahan rezim dilaporkan. |
| `domain/wilayah` | 152 | snippet:domain/wilayah | Pencocokan/fuzzy wilayah bukan lokasi kejadian; kamus dan peta tetap referensi, bukan 152 baris wajib. |
| `eda/class_keywords` | 22 | snippet:eda/class_keywords | Kata pembeda dapat artefak; classifier kata kunci tidak membuktikan asal anotasi label. |
| `eda/eda_report` | 101 | snippet:eda/eda_report | 101 baris helper diganti empat sensus literal; penyebut, sampel, hapax dan noise terlihat langsung. |
| `eda/label_distribution` | 22 | snippet:eda/label_distribution | Unknown multilabel perlu penyebut diketahui; biner, multiclass, numeric dan ABSA tidak ditangani dengan rumus sama. |
| `eda/length_stats` | 19 | snippet:eda/length_stats | Persentil kata bukan token model; cutoff BERT dari tokenizer, bukan rata-rata karakter. |
| `eda/quick_profile` | 22 | snippet:eda/quick_profile | Missing/blank/pseudo-null dibedakan; tidak otomatis membuang kata NA atau memodifikasi test. |
| `eda/vocab_stats` | 25 | snippet:eda/vocab_stats | Hapax frekuensi token berbeda dari document-frequency; batas sampel ditulis agar pembaca tidak mengira sensus penuh. |
| `evaluation/error_analysis` | 39 | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| `evaluation/metrics_classification` | 27 | snippet:evaluation/metrics_classification | Kode lama mengasumsikan label integer 0..K-1; kartu baru memakai label asli dan mapping m.classes_. |
| `evaluation/metrics_clustering` | 24 | snippet:evaluation/metrics_clustering | Hungarian accuracy hanya saat ada gold dan definisinya cocok; internal metric bukan accuracy. |
| `evaluation/metrics_ner` | 46 | snippet:evaluation/metrics_ner | Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat. |
| `evaluation/metrics_qa` | 27 | snippet:evaluation/metrics_qa | Multi-reference ambil skor terbaik; normalisasi evaluator Indonesia tidak otomatis sama dengan SQuAD Inggris. |
| `evaluation/metrics_regression` | 19 | snippet:evaluation/metrics_regression | RMSE API terbaru, MAE median, korelasi konstan undefined; QWK perlu ordinal dan cutpoint development. |
| `evaluation/metrics_retrieval` | 29 | snippet:evaluation/metrics_retrieval | Cutoff MAP/MRR lama bukan @K; kartu baru cutoff eksplisit, multi-positive dan graded gain dipisah. |
| `evaluation/metrics_rouge` | 32 | snippet:evaluation/metrics_rouge | ROUGE sederhana tetap referensi; tokenisasi/stemming tidak setara evaluator resmi dan fakta perlu diaudit. |
| `evaluation/metrics_seq2seq` | 29 | snippet:evaluation/metrics_seq2seq | CER/WER dapat >1; BLEU smoothing/tokenisasi buatan sendiri tidak boleh diberi label skor resmi. |
| `evaluation/metrics_topic` | 35 | snippet:evaluation/metrics_topic | NPMI kasus p=1/term tak muncul perlu guard; coherence/diversity hanya diagnostik dan contoh manusia tetap wajib. |
| `evaluation/noise_floor` | 34 | snippet:evaluation/noise_floor | Bootstrap 2SE adalah ketidakpastian estimator, bukan label-noise floor; group bootstrap jika data berkelompok. |
| `feature_engineering/count_bow` | 7 | snippet:feature_engineering/count_bow | Counts tetap alternatif NB/LDA; batas vocabulary dan vocabulary train-only harus eksplisit. |
| `feature_engineering/doc2vec_train` | 20 | snippet:feature_engineering/doc2vec_train | Doc2Vec inference stokastik dan corpus kecil tidak stabil; dipindah dari jalur ketik inti ke referensi. |
| `feature_engineering/fasttext_train` | 28 | snippet:feature_engineering/fasttext_train | FastText dapat membantu subword/OOV tetapi bukan wajib; train embedding pada train fold saja. |
| `feature_engineering/graph_features` | 20 | snippet:feature_engineering/graph_features | Graph yang dibangun dari test/pasangan yang dinilai transduktif; gunakan graph untuk grouping, bukan leakage fitur tersembunyi. |
| `feature_engineering/hashing` | 8 | snippet:feature_engineering/hashing | Hashing berguna skala besar, collision dan sign memengaruhi NB; TF-IDF lebih jelas untuk baseline dan audit kata. |
| `feature_engineering/make_vectorizer` | 94 | snippet:feature_engineering/make_vectorizer | Factory 94 baris dan cache global terlalu berat untuk ditulis; tiap representasi menjadi sel eksplisit. |
| `feature_engineering/meta_features` | 34 | snippet:feature_engineering/meta_features | Fitur volume seluruh hari dapat melihat masa depan; jangan membocorkan sumber/test lewat meta feature tanpa audit shift. |
| `feature_engineering/nb_log_ratio` | 18 | snippet:feature_engineering/nb_log_ratio | NB-ratio lama hardcode y==1; biner positif harus didefinisikan, tidak default untuk multiclass. |
| `feature_engineering/pair_features` | 52 | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| `feature_engineering/sbert_encode` | 25 | snippet:feature_engineering/sbert_encode | Cache sebelumnya tidak mengikat data/prefix/urutan; checksum ditambahkan, kartu pendek tidak memakai cache tersembunyi. |
| `feature_engineering/svd_reduce` | 22 | snippet:feature_engineering/svd_reduce | Batasi dim oleh ukuran train; UMAP/t-SNE untuk visual sampel bukan generalisasi dan hindari densify besar. |
| `feature_engineering/tfidf_char` | 9 | snippet:feature_engineering/tfidf_char | Default 500k fitur terlalu besar untuk baseline cepat; cap 60k dan min_df=1 untuk data kecil/emoji. |
| `feature_engineering/tfidf_union` | 11 | snippet:feature_engineering/tfidf_union | Default total 800k fitur bukan titik awal lomba; cap masing-masing 40k dan bandingkan baseline word dahulu. |
| `feature_engineering/tfidf_word` | 9 | snippet:feature_engineering/tfidf_word | min_df=2/max_df=.95 bisa empty vocabulary pada data kecil; default pendek min_df=1 dengan peringatan emoji/blank. |
| `feature_engineering/transformer_embed` | 23 | snippet:feature_engineering/transformer_embed | Mask padding benar wajib; base BERT mean pool tidak otomatis semantic embedding berkualitas, pilih E5 dengan prefix. |
| `feature_engineering/word2vec_train` | 32 | snippet:feature_engineering/word2vec_train | 32 baris sklearn wrapper dipangkas; OOV nol terlihat, workers=1 untuk reproducibility yang lebih baik. |
| `imbalance/class_weights` | 11 | snippet:imbalance/class_weights | Bobot dihitung dari train saja dan urut label konsisten; balanced bukan jaminan menang atau kalibrasi baik. |
| `imbalance/focal_loss` | 14 | snippet:imbalance/focal_loss | Focal loss opsional setelah baseline; bobot dan gamma belum tentu cocok dataset lain, tidak wajib diketik. |
| `imbalance/resampling` | 22 | snippet:imbalance/resampling | Resampling hanya train; koreksi prior butuh prior target yang sah, tidak boleh ditebak dari label test. |
| `imbalance/smote_svd` | 11 | snippet:imbalance/smote_svd | SMOTE sparse mentah dan kelas minor <=k bermasalah; bobot kelas lebih sederhana sebelum interpolasi embedding. |
| `imbalance/threshold_tuning` | 21 | snippet:imbalance/threshold_tuning | Smoothing kurva dan nested row CV tidak otomatis menghormati grup; kartu memakai subset tuning/check per grup. |
| `modeling/absa_pair_input` | 15 | snippet:modeling/absa_pair_input | Long format harus dibuat setelah split komentar; missing unknown berbeda dari aspek tidak ada. |
| `modeling/absa_per_aspect` | 34 | snippet:modeling/absa_per_aspect | Satu model per aspek, label kategori dan missing dimasking; kelas train konstan dilaporkan. |
| `modeling/anomaly_text` | 40 | snippet:modeling/anomaly_text | Anomali bukan label noise; threshold/top-k perlu audit gold dan jangan otomatis menghapus titik langka. |
| `modeling/attention_rnn` | 18 | snippet:modeling/attention_rnn | Arsitektur neural custom membawa trainer/tokenizer tambahan; referensi lanjutan, bukan jalur ketik pertama. |
| `modeling/augment_text` | 34 | snippet:modeling/augment_text | Word swap/dropout tidak menjamin label terjaga, terutama negasi/angka/NLI; hanya train dan validasi dampaknya. |
| `modeling/cluster_hdbscan` | 24 | snippet:modeling/cluster_hdbscan | Noise -1 tidak wajib dipaksa KNN ke cluster; HDBSCAN butuh skala/dimensi dan audit outlier. |
| `modeling/cluster_kmeans` | 28 | snippet:modeling/cluster_kmeans | Pencarian k dibatasi development dan silhouette sampel; consensus bukan bukti cluster semantik benar. |
| `modeling/cnn_text` | 16 | snippet:modeling/cnn_text | CNN custom memerlukan kosakata/trainer/padding; tidak worth mengetik sebelum baseline linear selesai. |
| `modeling/hashing_sgd` | 18 | snippet:modeling/hashing_sgd | Untuk data sangat besar gunakan streaming SGDC loss log_loss; urutan kelas partial_fit harus konsisten. |
| `modeling/head_on_embed` | 39 | snippet:modeling/head_on_embed | Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain. |
| `modeling/keyword_extract` | 51 | snippet:modeling/keyword_extract | Ambil sparse top terms tanpa densify corpus; MMR diversity bukan metrik relevansi gold. |
| `modeling/lang_detect` | 24 | snippet:modeling/lang_detect | Language marker dan pretrained language-ID bukan gold, code-mixing/teks pendek belum tentu terdeteksi benar. |
| `modeling/linear_models` | 24 | snippet:modeling/linear_models | LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend. |
| `modeling/make_model` | 118 | snippet:modeling/make_model | Factory 118 baris dan arsitektur raw-text tidak membantu ketik ulang; impor estimator langsung. |
| `modeling/multilabel_ovr` | 13 | snippet:modeling/multilabel_ovr | OneVsRest biasa tidak memasking partial label; head per label train-known diperlukan. |
| `modeling/nb_complement` | 8 | snippet:modeling/nb_complement | NB hanya fitur nonnegatif; boleh kandidat counts/TF-IDF, bukan SVD negatif atau semua representasi. |
| `modeling/ner_baseline` | 42 | snippet:modeling/ner_baseline | CRF fitur token dibuka langsung; gazetteer dan majority token bukan pengganti exact-span evaluation. |
| `modeling/ner_transformer` | 63 | snippet:modeling/ner_transformer | Truncation lama mengisi O pada token hilang; audit coverage/stride, jangan mengklaim full tagging setelah crop. |
| `modeling/ngram_next_word` | 28 | snippet:modeling/ngram_next_word | Split dokumen sebelum membuat prefix; input evaluasi tidak boleh mengandung kata jawaban. |
| `modeling/nn_train` | 49 | snippet:modeling/nn_train | Custom training loop 49 baris ditambah tokenizer/arsitektur terlalu banyak; advanced reference saja. |
| `modeling/pair_biencoder` | 8 | snippet:modeling/pair_biencoder | Cosine membutuhkan embedding/prefix sesuai task; similarity bukan entailment probability. |
| `modeling/pair_cross_encoder` | 11 | snippet:modeling/pair_cross_encoder | Pair tokenizer dua input lebih tepat daripada teks SEP manual; simetri tidak diterapkan pada NLI. |
| `modeling/qa_extractive` | 20 | snippet:modeling/qa_extractive | Sentence retrieval baseline bukan span answer akurat; jawaban-tail regex tidak menjamin jawaban benar. |
| `modeling/qa_transformer` | 59 | snippet:modeling/qa_transformer | Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi. |
| `modeling/random_forest_svd` | 8 | snippet:modeling/random_forest_svd | RF di sparse vocabulary besar bukan default cepat; SVD dibatasi dan fit train, jangan densify tanpa ukur RAM. |
| `modeling/ranker_lgbm` | 14 | snippet:modeling/ranker_lgbm | Fungsi lama np.asarray(X) tidak cocok sparse; kartu baru menjaga sparse dan group size sesuai urutan qid. |
| `modeling/regression_linear` | 21 | snippet:modeling/regression_linear | Regresi RMSE/MAE/RMSLE/ordinal berbeda; transform log tidak otomatis tepat dan nonnegative harus diperiksa. |
| `modeling/retrieval_bm25` | 27 | snippet:modeling/retrieval_bm25 | Tokenisasi dan ranking ID dijelaskan; top-k per query menghindari query×corpus dense. |
| `modeling/retrieval_dense` | 37 | snippet:modeling/retrieval_dense | Chunk aggregation per doc dan self-match perlu perhatian; prefix/query-document berbeda. |
| `modeling/rnn_text` | 20 | snippet:modeling/rnn_text | RNN dari nol memerlukan trainer/kosakata; simpan sebagai referensi, tidak tampil sebagai sel inti cheatsheet. |
| `modeling/seq2seq` | 29 | snippet:modeling/seq2seq | Base model bukan translator/summarizer siap pakai; prefix dan language target bergantung model yang benar. |
| `modeling/setfit` | 17 | snippet:modeling/setfit | Embedding+LR lebih sedikit kode; SetFit optional data sedikit, tidak otomatis lebih baik dari baseline. |
| `modeling/summarize_extractive` | 29 | snippet:modeling/summarize_extractive | Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang. |
| `modeling/topic_embedding` | 36 | snippet:modeling/topic_embedding | Fit topik di sampel awal/unique train, transform sisanya; topik klaster tanpa gold tidak punya accuracy. |
| `modeling/topic_nmf_lda` | 24 | snippet:modeling/topic_nmf_lda | LDA memakai counts; NMF TF-IDF; k dibatasi rank data dan contoh topik harus diaudit. |
| `modeling/transformer_multilabel` | 24 | snippet:modeling/transformer_multilabel | Masked BCE ditambahkan untuk unknown; sigmoid independen, bukan softmax/argmax dan bukan label float otomatis cukup. |
| `modeling/transformer_regression` | 13 | snippet:modeling/transformer_regression | num_labels=1 dan float target; STS simetris berbeda dari kelas dan skala skor harus jelas. |
| `modeling/transformer_trainer` | 86 | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| `modeling/tree_sparse` | 18 | snippet:modeling/tree_sparse | Tree tidak selalu lebih baik pada TF-IDF; objective/class count dan early stopping perlu schema yang benar. |
| `modeling/zero_shot` | 30 | snippet:modeling/zero_shot | NLI zero-shot memberi label lemah, bukan gold; multi score membandingkan entail/contradiction, neutral diabaikan. |
| `noisy_data/clean_all` | 58 | snippet:noisy_data/clean_all | Default majority-label correction sebelum split dihapus; dedup sekarang mempertahankan label konflik dalam grup sama. |
| `noisy_data/conflicting_labels` | 14 | snippet:noisy_data/conflicting_labels | Konflik ditampilkan dan dikelompokkan, tidak otomatis diputuskan ke label mayoritas sebelum evaluasi. |
| `noisy_data/duplicates` | 19 | snippet:noisy_data/duplicates | Normalisasi regex yang membuang tanda baca dapat menyatukan negasi/angka berbeda; grouping minimal lebih konservatif. |
| `noisy_data/label_noise_oof` | 16 | snippet:noisy_data/label_noise_oof | Cleanlab/low confidence hanya prioritas audit, bukan bukti salah anotasi atau alasan drop otomatis. |
| `noisy_data/text_outliers` | 25 | snippet:noisy_data/text_outliers | Noise regex deskriptif; emoji/non-ASCII/dialek tidak otomatis data buruk. |
| `optimization/amp_grad_accum` | 45 | snippet:optimization/amp_grad_accum | Manual loop/LLRD kompleks bukan wajib; fp16/accumulation jangan menurunkan learning rate tanpa hitung ulang langkah. |
| `optimization/encode_cached` | 16 | snippet:optimization/encode_cached | Memmap perlu metadata jumlah/urutan/model/prefix; jangan menerima cache hanya karena file ada. |
| `optimization/gpu_check` | 23 | snippet:optimization/gpu_check | Periksa perangkat saat setup; runtime CPU fallback terlihat, RTX 3070 tidak diasumsikan mendukung bf16. |
| `optimization/oom_fallback` | 17 | snippet:optimization/oom_fallback | Retry model training di state yang sudah setengah berubah dapat mengulang langkah; simpan checkpoint, kecilkan batch eksplisit. |
| `optimization/race_plan` | 40 | snippet:optimization/race_plan | Gate/scale heuristic bukan benchmark PC panitia; tidak membuang char/fold diam-diam hanya karena jumlah baris. |
| `optimization/save_artifacts` | 29 | snippet:optimization/save_artifacts | Nama artifact/order label/versi harus ada; penyimpanan joblib singkat cukup untuk baseline. |
| `optimization/seed_everything` | 11 | snippet:optimization/seed_everything | Seed bukan jaminan bitwise deterministic semua CUDA; workers/algoritme dan split tetap dicatat. |
| `optimization/subsample_stratified` | 15 | snippet:optimization/subsample_stratified | Sampel stratified tidak dapat menyelamatkan kelas cuma satu grup; jangan subsample lalu berbagi duplicate dengan validation. |
| `optimization/throughput_timer` | 31 | snippet:optimization/throughput_timer | Estimasi waktu dari batch pemanasan perlu margin; tidak menjanjikan semua pretrained selesai pada PC panitia. |
| `output/plots` | 42 | snippet:output/plots | Grafik sederhana biru/kuning; bukan donut/dekorasi wajib dan angka EDA tetap tersedia. |
| `output/slides` | 22 | snippet:output/slides | PPTX optional output; tugas cheatsheet tidak perlu mengetik helper slide sebelum jawaban/data/model benar. |
| `postprocessing/calibration` | 26 | snippet:postprocessing/calibration | Temperature tidak mengubah argmax; perbaikan log-loss bukan otomatis peningkatan F1. |
| `postprocessing/class_thresholds` | 41 | snippet:postprocessing/class_thresholds | Tuning/check grup dipisah; threshold rare label fallback dan tidak memaksa minimal satu label aktif. |
| `postprocessing/label_mapping` | 29 | snippet:postprocessing/label_mapping | ID dibaca string agar nol depan utuh; format sample_submission dan label order wajib cocok. |
| `postprocessing/ordinal_cutpoints` | 39 | snippet:postprocessing/ordinal_cutpoints | Ordinal cutpoint memerlukan check independen dan kelas terurut; jangan menamai regresi biasa sebagai QWK optimal. |
| `postprocessing/prob_average` | 9 | snippet:postprocessing/prob_average | Ensemble hanya pada baris dan label order sama; logits/margin tidak dicampur sebagai probability. |
| `postprocessing/rank_average` | 9 | snippet:postprocessing/rank_average | Rank average untuk ranking, bukan probability/log-loss; kandidat/cutoff dan group query harus sama. |
| `postprocessing/rule_overrides` | 30 | snippet:postprocessing/rule_overrides | Lookup train atau regex hanya sesuai aturan dan audit train/gold; matching yang menghapus tanda baca tidak membuktikan identitas semantik. |
| `postprocessing/stacking` | 18 | snippet:postprocessing/stacking | OOF meta row CV tidak otomatis jujur per grup; nested base predictions perlu untuk estimasi ketat. Blend sederhana dulu. |
| `postprocessing/tta` | 19 | snippet:postprocessing/tta | Head/tail/chunks bukan augmentasi valid semua task; QA offset dan NLI direction tidak boleh dirusak. |
| `preprocessing/branch_advanced` | 25 | snippet:preprocessing/branch_advanced | Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset. |
| `preprocessing/branch_minimal` | 13 | snippet:preprocessing/branch_minimal | NFC dan whitespace eksplisit; NFKC/casefold dapat mengubah entitas dan offset, tidak otomatis aman semua task. |
| `preprocessing/branch_traditional` | 17 | snippet:preprocessing/branch_traditional | Stemming/stopwords bukan wajib classifier; negasi/daerah dijaga dan uji pada development. |
| `preprocessing/branch_transformer` | 27 | snippet:preprocessing/branch_transformer | Native dan BERTweet dipisah; pasangan diberikan lewat tokenizer dua input, bukan SEP literal universal. |
| `preprocessing/preprocess_all` | 98 | snippet:preprocessing/preprocess_all | 98 baris saklar diganti cleaning kecil + normalisasi pilihan; kode yang diketik terlihat seluruhnya. |
| `preprocessing/slang_id` | 28 | snippet:preprocessing/slang_id | Kamus formal-domain spesifik, audit train; mapping berbatas kata, tidak substring replacement. |
| `preprocessing/stopwords_id` | 14 | snippet:preprocessing/stopwords_id | Negasi tetap dipertahankan; stopwords dipakai bila cocok interpretasi topik, bukan default classifier. |
| `semi_supervised/confidence_filter` | 28 | snippet:semi_supervised/confidence_filter | Confidence tinggi belum tentu benar/terkalibrasi; cek coverage per kelas dan gold tidak ikut pool. |
| `semi_supervised/iterative_pseudo` | 25 | snippet:semi_supervised/iterative_pseudo | Iterasi memilih best pada gold menjadikannya development; tetap perlu final gold independen untuk klaim. |
| `semi_supervised/pseudo_label` | 18 | snippet:semi_supervised/pseudo_label | Fit ulang vocabulary+classifier tidak memasukkan validation; pool duplicate/penulis lintas gold diaudit. |
| `semi_supervised/self_training` | 10 | snippet:semi_supervised/self_training | Label sentinel -1 hanya untuk unlabeled task tunggal; model harus predict_proba dan gold validation tetap terpisah. |
| `semi_supervised/weak_supervision` | 28 | snippet:semi_supervised/weak_supervision | Voting konflik lama memaksa argmax/tie; kartu baru abstain saat multi-hit/no-hit dan audit coverage/precision. |
| `validation/kfold_group` | 12 | snippet:validation/kfold_group | Unit independen ditetapkan dari data, bukan sekadar random group; jumlah grup dan kelas tiap fold dicek. |
| `validation/kfold_strat_group` | 8 | snippet:validation/kfold_strat_group | SGKF hanya mendekati stratifikasi di bawah batas grup; proporsi dan missing class tetap diperiksa. |
| `validation/kfold_stratified` | 17 | snippet:validation/kfold_stratified | Stratify tidak menghentikan leakage duplicate dan gagal pada rare class; gunakan grouping/time sesuai test. |
| `validation/leakage_overlap` | 17 | snippet:validation/leakage_overlap | Overlap normalized text bukan satu-satunya leakage; dua endpoint/user/near-duplicate memerlukan grup sendiri. |
| `validation/make_folds` | 101 | snippet:validation/make_folds | Multilabel groups sebelumnya diabaikan, NaN cast bool aktif; diperbaiki. OOF margin palsu probability kini ditolak. |
| `validation/multilabel_split` | 31 | snippet:validation/multilabel_split | NaN dahulu menjadi True; kini positive/known profile. Split rows tanpa grup belum aman terhadap duplicate. |
| `validation/near_dup_groups` | 46 | snippet:validation/near_dup_groups | Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama. |
| `validation/oof_predict` | 44 | snippet:validation/oof_predict | OOF belum terisi kini NaN, kolom kelas disejajarkan; sigmoid/softmax margin tidak diterima sebagai probability. |
| `validation/split_holdout` | 13 | snippet:validation/split_holdout | Stratified rows saja bocor pada teks/pengguna berulang; default grouped holdout dengan assert keterpisahan. |
| `validation/split_time` | 17 | snippet:validation/split_time | Timestamp invalid/tie/timezone dan recurrence harus dilaporkan; jangan memecah tanggal sama demi ukuran exact. |

## Saran metode satu per satu

| Metode | Rujukan | Kartu pendek | Catatan |
|---|---|---|---|
| CSV/TSV ke Parquet | data_loading/load_table | snippet:data_loading/load_table | Cache parquet bisa kedaluwarsa dan on_bad_lines bisa kehilangan baris; pembacaan eksplisit lebih mudah diaudit. |
| Coba beberapa encoding | data_loading/load_table | snippet:data_loading/load_table | Cache parquet bisa kedaluwarsa dan on_bad_lines bisa kehilangan baris; pembacaan eksplisit lebih mudah diaudit. |
| JSONL dan JSON | data_loading/load_table | snippet:data_loading/load_table | Cache parquet bisa kedaluwarsa dan on_bad_lines bisa kehilangan baris; pembacaan eksplisit lebih mudah diaudit. |
| Polars untuk file besar | data_loading/load_table | snippet:data_loading/load_table | Cache parquet bisa kedaluwarsa dan on_bad_lines bisa kehilangan baris; pembacaan eksplisit lebih mudah diaudit. |
| CoNLL/BIO | data_loading/load_conll | snippet:data_loading/load_conll | Parser CoNLL tetap referensi format khusus; jangan hilangkan batas dokumen atau mengubah offset. |
| Format offset span | data_loading/load_conll | snippet:data_loading/load_conll | Parser CoNLL tetap referensi format khusus; jangan hilangkan batas dokumen atau mengubah offset. |
| Format konteks-pertanyaan-jawaban | modeling/qa_extractive | snippet:modeling/qa_extractive | Sentence retrieval baseline bukan span answer akurat; jawaban-tail regex tidak menjamin jawaban benar. |
| Tebak kolom id, teks, target | data_loading/detect_columns | snippet:data_loading/detect_columns | Tebakan ID/target dari kardinalitas tidak boleh menentukan schema otomatis; tetapkan kolom dari soal. |
| Hemat memori | data_loading/reduce_memory | snippet:data_loading/reduce_memory | Downcast dapat mengubah presisi; category tidak wajib untuk teks, ukur RAM sebelum menyalin DataFrame. |
| Profil kolom | eda/quick_profile | snippet:eda/quick_profile | Missing/blank/pseudo-null dibedakan; tidak otomatis membuang kata NA atau memodifikasi test. |
| Baca 50 teks mentah | eda/quick_profile | snippet:eda/quick_profile | Missing/blank/pseudo-null dibedakan; tidak otomatis membuang kata NA atau memodifikasi test. |
| Kolom waktu dan sumber | eda/quick_profile | snippet:eda/quick_profile | Missing/blank/pseudo-null dibedakan; tidak otomatis membuang kata NA atau memodifikasi test. |
| Distribusi kelas | eda/label_distribution | snippet:eda/label_distribution | Unknown multilabel perlu penyebut diketahui; biner, multiclass, numeric dan ABSA tidak ditangani dengan rumus sama. |
| Kardinalitas dan co-occurrence label | eda/label_distribution | snippet:eda/label_distribution | Unknown multilabel perlu penyebut diketahui; biner, multiclass, numeric dan ABSA tidak ditangani dengan rumus sama. |
| Matriks aspek × sentimen | eda/label_distribution | snippet:eda/label_distribution | Unknown multilabel perlu penyebut diketahui; biner, multiclass, numeric dan ABSA tidak ditangani dengan rumus sama. |
| Histogram dan skew target | eda/label_distribution | snippet:eda/label_distribution | Unknown multilabel perlu penyebut diketahui; biner, multiclass, numeric dan ABSA tidak ditangani dengan rumus sama. |
| Uji asal label | eda/class_keywords | snippet:eda/class_keywords | Kata pembeda dapat artefak; classifier kata kunci tidak membuktikan asal anotasi label. |
| Persentil panjang | eda/length_stats | snippet:eda/length_stats | Persentil kata bukan token model; cutoff BERT dari tokenizer, bukan rata-rata karakter. |
| Panjang per kelas | eda/length_stats | snippet:eda/length_stats | Persentil kata bukan token model; cutoff BERT dari tokenizer, bukan rata-rata karakter. |
| Konteks dan jawaban | eda/length_stats | snippet:eda/length_stats | Persentil kata bukan token model; cutoff BERT dari tokenizer, bukan rata-rata karakter. |
| Vocabulary dan kata langka | eda/vocab_stats | snippet:eda/vocab_stats | Hapax frekuensi token berbeda dari document-frequency; batas sampel ditulis agar pembaca tidak mengira sensus penuh. |
| OOV test terhadap train | eda/vocab_stats | snippet:eda/vocab_stats | Hapax frekuensi token berbeda dari document-frequency; batas sampel ditulis agar pembaca tidak mengira sensus penuh. |
| Kata khas per kelas | eda/class_keywords | snippet:eda/class_keywords | Kata pembeda dapat artefak; classifier kata kunci tidak membuktikan asal anotasi label. |
| Kata teratas dan n-gram | eda/vocab_stats | snippet:eda/vocab_stats | Hapax frekuensi token berbeda dari document-frequency; batas sampel ditulis agar pembaca tidak mengira sensus penuh. |
| Deteksi bahasa dan kata penanda | modeling/lang_detect | snippet:modeling/lang_detect | Language marker dan pretrained language-ID bukan gold, code-mixing/teks pendek belum tentu terdeteksi benar. |
| Baris tak bermakna | noisy_data/text_outliers | snippet:noisy_data/text_outliers | Noise regex deskriptif; emoji/non-ASCII/dialek tidak otomatis data buruk. |
| Pola angka, URL, hashtag | feature_engineering/meta_features | snippet:feature_engineering/meta_features | Fitur volume seluruh hari dapat melihat masa depan; jangan membocorkan sumber/test lewat meta feature tanpa audit shift. |
| Lonjakan volume dan garis waktu | distribution_shift/feature_shift | snippet:distribution_shift/feature_shift | PSI konstan/kosong dan smoothing centered perlu perhatian; waktu online tidak boleh memakai periode masa depan. |
| Copypasta dan template | noisy_data/duplicates | snippet:noisy_data/duplicates | Normalisasi regex yang membuang tanda baca dapat menyatukan negasi/angka berbeda; grouping minimal lebih konservatif. |
| Cara test diambil | validation/leakage_overlap | snippet:validation/leakage_overlap | Overlap normalized text bukan satu-satunya leakage; dua endpoint/user/near-duplicate memerlukan grup sendiri. |
| Adversarial validation | distribution_shift/adversarial_validation | snippet:distribution_shift/adversarial_validation | Vectorizer lama fit gabungan sebelum CV domain; diagnostik transduktif bukan estimasi generalisasi induktif. |
| Vocabulary shift | distribution_shift/vocab_shift | snippet:distribution_shift/vocab_shift | Log-ratio vocabulary hanya deskriptif; jangan otomatis menghapus kata khas test atau mengoptimasi dari label test. |
| Length shift dan uji KS | distribution_shift/length_shift | snippet:distribution_shift/length_shift | KS p-value dipengaruhi jumlah data; laporkan besar shift/persentil dan jangan menganggap p kecil bukti dampak model. |
| PSI per fitur | distribution_shift/feature_shift | snippet:distribution_shift/feature_shift | PSI konstan/kosong dan smoothing centered perlu perhatian; waktu online tidak boleh memakai periode masa depan. |
| Overlap pasangan | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Graf teks berulang | validation/near_dup_groups | snippet:validation/near_dup_groups | Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama. |
| Relevansi per query | evaluation/metrics_retrieval | snippet:evaluation/metrics_retrieval | Cutoff MAP/MRR lama bukan @K; kartu baru cutoff eksplisit, multi-positive dan graded gain dipisah. |
| Sensus tipe dan panjang entitas | data_loading/load_conll | snippet:data_loading/load_conll | Parser CoNLL tetap referensi format khusus; jangan hilangkan batas dokumen atau mengubah offset. |
| Sensus tipe pertanyaan | modeling/qa_extractive | snippet:modeling/qa_extractive | Sentence retrieval baseline bukan span answer akurat; jawaban-tail regex tidak menjamin jawaban benar. |
| Rasio ringkasan | modeling/summarize_extractive | snippet:modeling/summarize_extractive | Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang. |
| Duplikat persis ternormalisasi | noisy_data/duplicates | snippet:noisy_data/duplicates | Normalisasi regex yang membuang tanda baca dapat menyatukan negasi/angka berbeda; grouping minimal lebih konservatif. |
| Near-duplicate dengan MinHash LSH | validation/near_dup_groups | snippet:validation/near_dup_groups | Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama. |
| Kemiripan fuzzy | noisy_data/duplicates | snippet:noisy_data/duplicates | Normalisasi regex yang membuang tanda baca dapat menyatukan negasi/angka berbeda; grouping minimal lebih konservatif. |
| Grup untuk validasi | validation/near_dup_groups | snippet:validation/near_dup_groups | Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama. |
| Conflicting labels | noisy_data/conflicting_labels | snippet:noisy_data/conflicting_labels | Konflik ditampilkan dan dikelompokkan, tidak otomatis diputuskan ke label mayoritas sebelum evaluasi. |
| Prioritas baca dengan cleanlab | noisy_data/label_noise_oof | snippet:noisy_data/label_noise_oof | Cleanlab/low confidence hanya prioritas audit, bukan bukti salah anotasi atau alasan drop otomatis. |
| Baca kesalahan paling yakin | evaluation/error_analysis | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| Gold kecil buatan sendiri | semi_supervised/weak_supervision | snippet:semi_supervised/weak_supervision | Voting konflik lama memaksa argmax/tie; kartu baru abstain saat multi-hit/no-hit dan audit coverage/precision. |
| Teks kosong dan pseudo-null | eda/quick_profile | snippet:eda/quick_profile | Missing/blank/pseudo-null dibedakan; tidak otomatis membuang kata NA atau memodifikasi test. |
| Outlier panjang dan rasio non-huruf | noisy_data/text_outliers | snippet:noisy_data/text_outliers | Noise regex deskriptif; emoji/non-ASCII/dialek tidak otomatis data buruk. |
| Noise domain | noisy_data/text_outliers | snippet:noisy_data/text_outliers | Noise regex deskriptif; emoji/non-ASCII/dialek tidak otomatis data buruk. |
| Tanpa preprocessing dulu | preprocessing/branch_minimal | snippet:preprocessing/branch_minimal | NFC dan whitespace eksplisit; NFKC/casefold dapat mengubah entitas dan offset, tidak otomatis aman semua task. |
| Lowercase | preprocessing/branch_minimal | snippet:preprocessing/branch_minimal | NFC dan whitespace eksplisit; NFKC/casefold dapat mengubah entitas dan offset, tidak otomatis aman semua task. |
| Normalisasi whitespace | preprocessing/branch_minimal | snippet:preprocessing/branch_minimal | NFC dan whitespace eksplisit; NFKC/casefold dapat mengubah entitas dan offset, tidak otomatis aman semua task. |
| Unicode normalize | preprocessing/branch_minimal | snippet:preprocessing/branch_minimal | NFC dan whitespace eksplisit; NFKC/casefold dapat mengubah entitas dan offset, tidak otomatis aman semua task. |
| Tokenisasi | preprocessing/branch_traditional | snippet:preprocessing/branch_traditional | Stemming/stopwords bukan wajib classifier; negasi/daerah dijaga dan uji pada development. |
| Stopword removal | preprocessing/stopwords_id | snippet:preprocessing/stopwords_id | Negasi tetap dipertahankan; stopwords dipakai bila cocok interpretasi topik, bukan default classifier. |
| Stemming Sastrawi | preprocessing/branch_traditional | snippet:preprocessing/branch_traditional | Stemming/stopwords bukan wajib classifier; negasi/daerah dijaga dan uji pada development. |
| Lemmatization | preprocessing/branch_traditional | snippet:preprocessing/branch_traditional | Stemming/stopwords bukan wajib classifier; negasi/daerah dijaga dan uji pada development. |
| URL dan mention jadi token | preprocessing/branch_advanced | snippet:preprocessing/branch_advanced | Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset. |
| Hapus HTML | preprocessing/branch_advanced | snippet:preprocessing/branch_advanced | Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset. |
| Emoji jadi teks | preprocessing/branch_advanced | snippet:preprocessing/branch_advanced | Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset. |
| Normalisasi slang | preprocessing/slang_id | snippet:preprocessing/slang_id | Kamus formal-domain spesifik, audit train; mapping berbatas kata, tidak substring replacement. |
| Kurangi huruf berulang | preprocessing/branch_advanced | snippet:preprocessing/branch_advanced | Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset. |
| Normalisasi angka | preprocessing/branch_advanced | snippet:preprocessing/branch_advanced | Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset. |
| Pertahankan negasi | preprocessing/slang_id | snippet:preprocessing/slang_id | Kamus formal-domain spesifik, audit train; mapping berbatas kata, tidak substring replacement. |
| Preprocessing sesuai model | preprocessing/branch_transformer | snippet:preprocessing/branch_transformer | Native dan BERTweet dipisah; pasangan diberikan lewat tokenizer dua input, bukan SEP literal universal. |
| Tanpa stemming dan tanpa stopword | preprocessing/branch_transformer | snippet:preprocessing/branch_transformer | Native dan BERTweet dipisah; pasangan diberikan lewat tokenizer dua input, bukan SEP literal universal. |
| Teks gabungan bertanda | preprocessing/branch_transformer | snippet:preprocessing/branch_transformer | Native dan BERTweet dipisah; pasangan diberikan lewat tokenizer dua input, bukan SEP literal universal. |
| Prefiks E5 | feature_engineering/sbert_encode | snippet:feature_engineering/sbert_encode | Cache sebelumnya tidak mengikat data/prefix/urutan; checksum ditambahkan, kartu pendek tidak memakai cache tersembunyi. |
| Pisah kalimat | modeling/summarize_extractive | snippet:modeling/summarize_extractive | Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang. |
| Sliding window dokumen | modeling/qa_transformer | snippet:modeling/qa_transformer | Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi. |
| Tokenisasi tersejajar | modeling/ner_transformer | snippet:modeling/ner_transformer | Truncation lama mengisi O pada token hilang; audit coverage/stride, jangan mengklaim full tagging setelah crop. |
| TF-IDF kata | feature_engineering/tfidf_word | snippet:feature_engineering/tfidf_word | min_df=2/max_df=.95 bisa empty vocabulary pada data kecil; default pendek min_df=1 dengan peringatan emoji/blank. |
| TF-IDF karakter | feature_engineering/tfidf_char | snippet:feature_engineering/tfidf_char | Default 500k fitur terlalu besar untuk baseline cepat; cap 60k dan min_df=1 untuk data kecil/emoji. |
| Gabungan kata + karakter | feature_engineering/tfidf_union | snippet:feature_engineering/tfidf_union | Default total 800k fitur bukan titik awal lomba; cap masing-masing 40k dan bandingkan baseline word dahulu. |
| CountVectorizer | feature_engineering/count_bow | snippet:feature_engineering/count_bow | Counts tetap alternatif NB/LDA; batas vocabulary dan vocabulary train-only harus eksplisit. |
| Binary BoW | feature_engineering/count_bow | snippet:feature_engineering/count_bow | Counts tetap alternatif NB/LDA; batas vocabulary dan vocabulary train-only harus eksplisit. |
| Hashing | feature_engineering/hashing | snippet:feature_engineering/hashing | Hashing berguna skala besar, collision dan sign memengaruhi NB; TF-IDF lebih jelas untuk baseline dan audit kata. |
| NB-SVM log-count ratio | feature_engineering/nb_log_ratio | snippet:feature_engineering/nb_log_ratio | NB-ratio lama hardcode y==1; biner positif harus didefinisikan, tidak default untuk multiclass. |
| TruncatedSVD (LSA) | feature_engineering/svd_reduce | snippet:feature_engineering/svd_reduce | Batasi dim oleh ukuran train; UMAP/t-SNE untuk visual sampel bukan generalisasi dan hindari densify besar. |
| IncrementalPCA | feature_engineering/svd_reduce | snippet:feature_engineering/svd_reduce | Batasi dim oleh ukuran train; UMAP/t-SNE untuk visual sampel bukan generalisasi dan hindari densify besar. |
| UMAP | modeling/topic_embedding | snippet:modeling/topic_embedding | Fit topik di sampel awal/unique train, transform sisanya; topik klaster tanpa gold tidak punya accuracy. |
| t-SNE | feature_engineering/svd_reduce | snippet:feature_engineering/svd_reduce | Batasi dim oleh ukuran train; UMAP/t-SNE untuk visual sampel bukan generalisasi dan hindari densify besar. |
| Word2Vec | feature_engineering/word2vec_train | snippet:feature_engineering/word2vec_train | 32 baris sklearn wrapper dipangkas; OOV nol terlihat, workers=1 untuk reproducibility yang lebih baik. |
| FastText sendiri | feature_engineering/fasttext_train | snippet:feature_engineering/fasttext_train | FastText dapat membantu subword/OOV tetapi bukan wajib; train embedding pada train fold saja. |
| Doc2Vec | feature_engineering/doc2vec_train | snippet:feature_engineering/doc2vec_train | Doc2Vec inference stokastik dan corpus kecil tidak stabil; dipindah dari jalur ketik inti ke referensi. |
| TF-IDF weighted embedding | feature_engineering/word2vec_train | snippet:feature_engineering/word2vec_train | 32 baris sklearn wrapper dipangkas; OOV nol terlihat, workers=1 untuk reproducibility yang lebih baik. |
| fastText cc.id.300.vec | feature_engineering/fasttext_train | snippet:feature_engineering/fasttext_train | FastText dapat membantu subword/OOV tetapi bukan wajib; train embedding pada train fold saja. |
| Sentence embedding (E5, MiniLM) | feature_engineering/sbert_encode | snippet:feature_engineering/sbert_encode | Cache sebelumnya tidak mengikat data/prefix/urutan; checksum ditambahkan, kartu pendek tidak memakai cache tersembunyi. |
| Mean pooling transformer | feature_engineering/transformer_embed | snippet:feature_engineering/transformer_embed | Mask padding benar wajib; base BERT mean pool tidak otomatis semantic embedding berkualitas, pilih E5 dengan prefix. |
| Skor model siap pakai sebagai fitur | modeling/head_on_embed | snippet:modeling/head_on_embed | Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain. |
| Meta features | feature_engineering/meta_features | snippet:feature_engineering/meta_features | Fitur volume seluruh hari dapat melihat masa depan; jangan membocorkan sumber/test lewat meta feature tanpa audit shift. |
| Fitur leksikon aspek | semi_supervised/weak_supervision | snippet:semi_supervised/weak_supervision | Voting konflik lama memaksa argmax/tie; kartu baru abstain saat multi-hit/no-hit dan audit coverage/precision. |
| InSet sebagai fitur | feature_engineering/meta_features | snippet:feature_engineering/meta_features | Fitur volume seluruh hari dapat melihat masa depan; jangan membocorkan sumber/test lewat meta feature tanpa audit shift. |
| Fitur waktu dan burst | feature_engineering/meta_features | snippet:feature_engineering/meta_features | Fitur volume seluruh hari dapat melihat masa depan; jangan membocorkan sumber/test lewat meta feature tanpa audit shift. |
| Frekuensi teks near-duplicate | validation/near_dup_groups | snippet:validation/near_dup_groups | Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama. |
| Overlap dan Jaccard | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Kosinus TF-IDF dan embedding | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Fuzzy ratio | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Konflik angka, entitas, negasi | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Jendela terbaik | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Fitur graf | feature_engineering/graph_features | snippet:feature_engineering/graph_features | Graph yang dibangun dari test/pasangan yang dinilai transduktif; gunakan graph untuk grouping, bukan leakage fitur tersembunyi. |
| Fitur bentuk kata | modeling/ner_baseline | snippet:modeling/ner_baseline | CRF fitur token dibuka langsung; gazetteer dan majority token bukan pengganti exact-span evaluation. |
| Gazetteer | modeling/ner_baseline | snippet:modeling/ner_baseline | CRF fitur token dibuka langsung; gazetteer dan majority token bukan pengganti exact-span evaluation. |
| Train-validation split | validation/split_holdout | snippet:validation/split_holdout | Stratified rows saja bocor pada teks/pengguna berulang; default grouped holdout dengan assert keterpisahan. |
| Stratified K-Fold | validation/kfold_stratified | snippet:validation/kfold_stratified | Stratify tidak menghentikan leakage duplicate dan gagal pada rare class; gunakan grouping/time sesuai test. |
| K-Fold | validation/kfold_stratified | snippet:validation/kfold_stratified | Stratify tidak menghentikan leakage duplicate dan gagal pada rare class; gunakan grouping/time sesuai test. |
| Group K-Fold | validation/kfold_group | snippet:validation/kfold_group | Unit independen ditetapkan dari data, bukan sekadar random group; jumlah grup dan kelas tiap fold dicek. |
| Stratified Group K-Fold | validation/kfold_strat_group | snippet:validation/kfold_strat_group | SGKF hanya mendekati stratifikasi di bawah batas grup; proporsi dan missing class tetap diperiksa. |
| Iterative stratification | validation/multilabel_split | snippet:validation/multilabel_split | NaN dahulu menjadi True; kini positive/known profile. Split rows tanpa grup belum aman terhadap duplicate. |
| Time split | validation/split_time | snippet:validation/split_time | Timestamp invalid/tie/timezone dan recurrence harus dilaporkan; jangan memecah tanggal sama demi ukuran exact. |
| Leave-one-source-out | validation/kfold_group | snippet:validation/kfold_group | Unit independen ditetapkan dari data, bukan sekadar random group; jumlah grup dan kelas tiap fold dicek. |
| Fold per query | validation/kfold_group | snippet:validation/kfold_group | Unit independen ditetapkan dari data, bukan sekadar random group; jumlah grup dan kelas tiap fold dicek. |
| Fold per komponen graf | validation/near_dup_groups | snippet:validation/near_dup_groups | Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama. |
| Fold per dokumen | validation/kfold_group | snippet:validation/kfold_group | Unit independen ditetapkan dari data, bukan sekadar random group; jumlah grup dan kelas tiap fold dicek. |
| Stabilitas antar seed | evaluation/metrics_clustering | snippet:evaluation/metrics_clustering | Hungarian accuracy hanya saat ada gold dan definisinya cocok; internal metric bukan accuracy. |
| Injeksi sintetis | modeling/anomaly_text | snippet:modeling/anomaly_text | Anomali bukan label noise; threshold/top-k perlu audit gold dan jangan otomatis menghapus titik langka. |
| Validasi hanya pada gold | semi_supervised/pseudo_label | snippet:semi_supervised/pseudo_label | Fit ulang vocabulary+classifier tidak memasukkan validation; pool duplicate/penulis lintas gold diaudit. |
| Cek overlap lintas split | validation/leakage_overlap | snippet:validation/leakage_overlap | Overlap normalized text bukan satu-satunya leakage; dua endpoint/user/near-duplicate memerlukan grup sendiri. |
| Pipeline dalam fold | validation/kfold_stratified | snippet:validation/kfold_stratified | Stratify tidak menghentikan leakage duplicate dan gagal pada rare class; gunakan grouping/time sesuai test. |
| Majority dan prior | modeling/linear_models | snippet:modeling/linear_models | LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend. |
| Mean dan median | modeling/regression_linear | snippet:modeling/regression_linear | Regresi RMSE/MAE/RMSLE/ordinal berbeda; transform log tidak otomatis tepat dan nonnegative harus diperiksa. |
| Logistic Regression | modeling/linear_models | snippet:modeling/linear_models | LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend. |
| LinearSVC | modeling/linear_models | snippet:modeling/linear_models | LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend. |
| SGD + partial_fit | modeling/hashing_sgd | snippet:modeling/hashing_sgd | Untuk data sangat besar gunakan streaming SGDC loss log_loss; urutan kelas partial_fit harus konsisten. |
| Ridge | modeling/regression_linear | snippet:modeling/regression_linear | Regresi RMSE/MAE/RMSLE/ordinal berbeda; transform log tidak otomatis tepat dan nonnegative harus diperiksa. |
| RidgeClassifier | modeling/linear_models | snippet:modeling/linear_models | LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend. |
| Naive Bayes | modeling/nb_complement | snippet:modeling/nb_complement | NB hanya fitur nonnegatif; boleh kandidat counts/TF-IDF, bukan SVD negatif atau semua representasi. |
| One-vs-Rest biner | modeling/multilabel_ovr | snippet:modeling/multilabel_ovr | OneVsRest biasa tidak memasking partial label; head per label train-known diperlukan. |
| Classifier Chain | modeling/multilabel_ovr | snippet:modeling/multilabel_ovr | OneVsRest biasa tidak memasking partial label; head per label train-known diperlukan. |
| LightGBM klasifikasi | modeling/tree_sparse | snippet:modeling/tree_sparse | Tree tidak selalu lebih baik pada TF-IDF; objective/class count dan early stopping perlu schema yang benar. |
| LightGBM regresi | modeling/regression_linear | snippet:modeling/regression_linear | Regresi RMSE/MAE/RMSLE/ordinal berbeda; transform log tidak otomatis tepat dan nonnegative harus diperiksa. |
| XGBoost | modeling/tree_sparse | snippet:modeling/tree_sparse | Tree tidak selalu lebih baik pada TF-IDF; objective/class count dan early stopping perlu schema yang benar. |
| Random Forest | modeling/random_forest_svd | snippet:modeling/random_forest_svd | RF di sparse vocabulary besar bukan default cepat; SVD dibatasi dan fit train, jangan densify tanpa ukur RAM. |
| TextCNN | modeling/cnn_text | snippet:modeling/cnn_text | CNN custom memerlukan kosakata/trainer/padding; tidak worth mengetik sebelum baseline linear selesai. |
| BiGRU / BiLSTM | modeling/rnn_text | snippet:modeling/rnn_text | RNN dari nol memerlukan trainer/kosakata; simpan sebagai referensi, tidak tampil sebagai sel inti cheatsheet. |
| Attention pooling | modeling/attention_rnn | snippet:modeling/attention_rnn | Arsitektur neural custom membawa trainer/tokenizer tambahan; referensi lanjutan, bukan jalur ketik pertama. |
| MLP di atas embedding | modeling/head_on_embed | snippet:modeling/head_on_embed | Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain. |
| LR di atas embedding | modeling/head_on_embed | snippet:modeling/head_on_embed | Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain. |
| SetFit | modeling/setfit | snippet:modeling/setfit | Embedding+LR lebih sedikit kode; SetFit optional data sedikit, tidak otomatis lebih baik dari baseline. |
| IndoBERTweet | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| NusaBERT | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| IndoBERT | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| XLM-RoBERTa | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Multilabel BCE | modeling/transformer_multilabel | snippet:modeling/transformer_multilabel | Masked BCE ditambahkan untuk unknown; sigmoid independen, bukan softmax/argmax dan bukan label float otomatis cukup. |
| Regresi transformer | modeling/transformer_regression | snippet:modeling/transformer_regression | num_labels=1 dan float target; STS simetris berbeda dari kelas dan skala skor harus jelas. |
| Fitur pasangan + LightGBM | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Bi-encoder + kosinus | modeling/pair_biencoder | snippet:modeling/pair_biencoder | Cosine membutuhkan embedding/prefix sesuai task; similarity bukan entailment probability. |
| Cross-encoder fine-tune | modeling/pair_cross_encoder | snippet:modeling/pair_cross_encoder | Pair tokenizer dua input lebih tepat daripada teks SEP manual; simetri tidak diterapkan pada NLI. |
| NLI siap pakai | modeling/zero_shot | snippet:modeling/zero_shot | NLI zero-shot memberi label lemah, bukan gold; multi score membandingkan entail/contradiction, neutral diabaikan. |
| Sentimen siap pakai | modeling/head_on_embed | snippet:modeling/head_on_embed | Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain. |
| Emosi siap pakai | modeling/head_on_embed | snippet:modeling/head_on_embed | Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain. |
| Zero-shot NLI | modeling/zero_shot | snippet:modeling/zero_shot | NLI zero-shot memberi label lemah, bukan gold; multi score membandingkan entail/contradiction, neutral diabaikan. |
| Hoaks siap pakai | model |  | Contoh API ilustratif, bukan sel mandiri. Gunakan input dan langkah eksplisit kartu terkait. |
| Input aspek [SEP] teks | modeling/absa_pair_input | snippet:modeling/absa_pair_input | Long format harus dibuat setelah split komentar; missing unknown berbeda dari aspek tidak ada. |
| Model per aspek | modeling/absa_per_aspect | snippet:modeling/absa_per_aspect | Satu model per aspek, label kategori dan missing dimasking; kelas train konstan dilaporkan. |
| Most-frequent-tag + aturan | modeling/ner_baseline | snippet:modeling/ner_baseline | CRF fitur token dibuka langsung; gazetteer dan majority token bukan pengganti exact-span evaluation. |
| NusaBert NER siap pakai | modeling/ner_transformer | snippet:modeling/ner_transformer | Truncation lama mengisi O pada token hilang; audit coverage/stride, jangan mengklaim full tagging setelah crop. |
| Token classification fine-tune | modeling/ner_transformer | snippet:modeling/ner_transformer | Truncation lama mengisi O pada token hilang; audit coverage/stride, jangan mengklaim full tagging setelah crop. |
| CRF fitur manual | modeling/ner_baseline | snippet:modeling/ner_baseline | CRF fitur token dibuka langsung; gazetteer dan majority token bukan pengganti exact-span evaluation. |
| Ekstraktif TF-IDF | modeling/qa_extractive | snippet:modeling/qa_extractive | Sentence retrieval baseline bukan span answer akurat; jawaban-tail regex tidak menjamin jawaban benar. |
| IndoBERT QA | modeling/qa_transformer | snippet:modeling/qa_transformer | Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi. |
| mDeBERTa SQuAD2 | modeling/qa_transformer | snippet:modeling/qa_transformer | Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi. |
| Retriever + reader | modeling/retrieval_bm25 | snippet:modeling/retrieval_bm25 | Tokenisasi dan ranking ID dijelaskan; top-k per query menghindari query×corpus dense. |
| BM25 sparse | modeling/retrieval_bm25 | snippet:modeling/retrieval_bm25 | Tokenisasi dan ranking ID dijelaskan; top-k per query menghindari query×corpus dense. |
| TF-IDF kosinus top-k | modeling/retrieval_dense | snippet:modeling/retrieval_dense | Chunk aggregation per doc dan self-match perlu perhatian; prefix/query-document berbeda. |
| Dense retrieval | modeling/retrieval_dense | snippet:modeling/retrieval_dense | Chunk aggregation per doc dan self-match perlu perhatian; prefix/query-document berbeda. |
| Hybrid | modeling/retrieval_dense | snippet:modeling/retrieval_dense | Chunk aggregation per doc dan self-match perlu perhatian; prefix/query-document berbeda. |
| LGBMRanker | modeling/ranker_lgbm | snippet:modeling/ranker_lgbm | Fungsi lama np.asarray(X) tidak cocok sparse; kartu baru menjaga sparse dan group size sesuai urutan qid. |
| Re-rank cross-encoder | modeling/pair_cross_encoder | snippet:modeling/pair_cross_encoder | Pair tokenizer dua input lebih tepat daripada teks SEP manual; simetri tidak diterapkan pada NLI. |
| NMF | modeling/topic_nmf_lda | snippet:modeling/topic_nmf_lda | LDA memakai counts; NMF TF-IDF; k dibatasi rank data dan contoh topik harus diaudit. |
| LDA | modeling/topic_nmf_lda | snippet:modeling/topic_nmf_lda | LDA memakai counts; NMF TF-IDF; k dibatasi rank data dan contoh topik harus diaudit. |
| BERTopic pada sampel | modeling/topic_embedding | snippet:modeling/topic_embedding | Fit topik di sampel awal/unique train, transform sisanya; topik klaster tanpa gold tidak punya accuracy. |
| Fallback MiniBatch topik | modeling/topic_embedding | snippet:modeling/topic_embedding | Fit topik di sampel awal/unique train, transform sisanya; topik klaster tanpa gold tidak punya accuracy. |
| Penamaan topik LLM | model |  | Contoh API ilustratif, bukan sel mandiri. Gunakan input dan langkah eksplisit kartu terkait. |
| MiniBatchKMeans | modeling/cluster_kmeans | snippet:modeling/cluster_kmeans | Pencarian k dibatasi development dan silhouette sampel; consensus bukan bukti cluster semantik benar. |
| HDBSCAN | modeling/cluster_hdbscan | snippet:modeling/cluster_hdbscan | Noise -1 tidak wajib dipaksa KNN ke cluster; HDBSCAN butuh skala/dimensi dan audit outlier. |
| Agglomerative | modeling/cluster_kmeans | snippet:modeling/cluster_kmeans | Pencarian k dibatasi development dan silhouette sampel; consensus bukan bukti cluster semantik benar. |
| Spherical KMeans (praktis) | modeling/cluster_kmeans | snippet:modeling/cluster_kmeans | Pencarian k dibatasi development dan silhouette sampel; consensus bukan bukti cluster semantik benar. |
| TF-IDF per dokumen | modeling/keyword_extract | snippet:modeling/keyword_extract | Ambil sparse top terms tanpa densify corpus; MMR diversity bukan metrik relevansi gold. |
| c-TF-IDF per kelompok | modeling/keyword_extract | snippet:modeling/keyword_extract | Ambil sparse top terms tanpa densify corpus; MMR diversity bukan metrik relevansi gold. |
| KeyBERT-style | modeling/keyword_extract | snippet:modeling/keyword_extract | Ambil sparse top terms tanpa densify corpus; MMR diversity bukan metrik relevansi gold. |
| Lead-N | modeling/summarize_extractive | snippet:modeling/summarize_extractive | Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang. |
| TextRank | modeling/summarize_extractive | snippet:modeling/summarize_extractive | Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang. |
| T5 Indonesia | modeling/seq2seq | snippet:modeling/seq2seq | Base model bukan translator/summarizer siap pakai; prefix dan language target bergantung model yang benar. |
| Char TF-IDF untuk bahasa | modeling/lang_detect | snippet:modeling/lang_detect | Language marker dan pretrained language-ID bukan gold, code-mixing/teks pendek belum tentu terdeteksi benar. |
| fast-langdetect | modeling/lang_detect | snippet:modeling/lang_detect | Language marker dan pretrained language-ID bukan gold, code-mixing/teks pendek belum tentu terdeteksi benar. |
| N-gram + backoff | modeling/ngram_next_word | snippet:modeling/ngram_next_word | Split dokumen sebelum membuat prefix; input evaluasi tidak boleh mengandung kata jawaban. |
| GRU bahasa | modeling/rnn_text | snippet:modeling/rnn_text | RNN dari nol memerlukan trainer/kosakata; simpan sebagai referensi, tidak tampil sebagai sel inti cheatsheet. |
| IsolationForest di SVD | modeling/anomaly_text | snippet:modeling/anomaly_text | Anomali bukan label noise; threshold/top-k perlu audit gold dan jangan otomatis menghapus titik langka. |
| Jarak kNN embedding | modeling/anomaly_text | snippet:modeling/anomaly_text | Anomali bukan label noise; threshold/top-k perlu audit gold dan jangan otomatis menghapus titik langka. |
| Reconstruction error SVD | modeling/anomaly_text | snippet:modeling/anomaly_text | Anomali bukan label noise; threshold/top-k perlu audit gold dan jangan otomatis menghapus titik langka. |
| SelfTrainingClassifier | semi_supervised/self_training | snippet:semi_supervised/self_training | Label sentinel -1 hanya untuk unlabeled task tunggal; model harus predict_proba dan gold validation tetap terpisah. |
| Pseudo-label satu putaran | semi_supervised/pseudo_label | snippet:semi_supervised/pseudo_label | Fit ulang vocabulary+classifier tidak memasukkan validation; pool duplicate/penulis lintas gold diaudit. |
| Filter keyakinan per kelas | semi_supervised/confidence_filter | snippet:semi_supervised/confidence_filter | Confidence tinggi belum tentu benar/terkalibrasi; cek coverage per kelas dan gold tidak ikut pool. |
| Iterasi dengan penghentian | semi_supervised/iterative_pseudo | snippet:semi_supervised/iterative_pseudo | Iterasi memilih best pada gold menjadikannya development; tetap perlu final gold independen untuk klaim. |
| Labeling functions | semi_supervised/weak_supervision | snippet:semi_supervised/weak_supervision | Voting konflik lama memaksa argmax/tie; kartu baru abstain saat multi-hit/no-hit dan audit coverage/precision. |
| Mulai tanpa class weight | imbalance/threshold_tuning | snippet:imbalance/threshold_tuning | Smoothing kurva dan nested row CV tidak otomatis menghormati grup; kartu memakai subset tuning/check per grup. |
| Class weight | imbalance/class_weights | snippet:imbalance/class_weights | Bobot dihitung dari train saja dan urut label konsisten; balanced bukan jaminan menang atau kalibrasi baik. |
| Random undersampling | imbalance/resampling | snippet:imbalance/resampling | Resampling hanya train; koreksi prior butuh prior target yang sah, tidak boleh ditebak dari label test. |
| Random oversampling | imbalance/resampling | snippet:imbalance/resampling | Resampling hanya train; koreksi prior butuh prior target yang sah, tidak boleh ditebak dari label test. |
| SMOTE di fitur padat | imbalance/smote_svd | snippet:imbalance/smote_svd | SMOTE sparse mentah dan kelas minor <=k bermasalah; bobot kelas lebih sederhana sebelum interpolasi embedding. |
| Focal loss | imbalance/focal_loss | snippet:imbalance/focal_loss | Focal loss opsional setelah baseline; bobot dan gamma belum tentu cocok dataset lain, tidak wajib diketik. |
| Subsample stratified | optimization/subsample_stratified | snippet:optimization/subsample_stratified | Sampel stratified tidak dapat menyelamatkan kelas cuma satu grup; jangan subsample lalu berbagi duplicate dengan validation. |
| Ukur throughput | optimization/throughput_timer | snippet:optimization/throughput_timer | Estimasi waktu dari batch pemanasan perlu margin; tidak menjanjikan semua pretrained selesai pada PC panitia. |
| Kill rule BERT | optimization/throughput_timer | snippet:optimization/throughput_timer | Estimasi waktu dari batch pemanasan perlu margin; tidak menjanjikan semua pretrained selesai pada PC panitia. |
| `n_jobs` terbatas | optimization/gpu_check | snippet:optimization/gpu_check | Periksa perangkat saat setup; runtime CPU fallback terlihat, RTX 3070 tidak diasumsikan mendukung bf16. |
| Embedding ke memmap | optimization/encode_cached | snippet:optimization/encode_cached | Memmap perlu metadata jumlah/urutan/model/prefix; jangan menerima cache hanya karena file ada. |
| Max length dari histogram | eda/length_stats | snippet:eda/length_stats | Persentil kata bukan token model; cutoff BERT dari tokenizer, bukan rata-rata karakter. |
| Dynamic padding | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Length bucketing | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Mixed precision bf16 | optimization/amp_grad_accum | snippet:optimization/amp_grad_accum | Manual loop/LLRD kompleks bukan wajib; fp16/accumulation jangan menurunkan learning rate tanpa hitung ulang langkah. |
| Gradient accumulation | optimization/amp_grad_accum | snippet:optimization/amp_grad_accum | Manual loop/LLRD kompleks bukan wajib; fp16/accumulation jangan menurunkan learning rate tanpa hitung ulang langkah. |
| Argumen training dasar | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Early stopping + resume | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Validasi sering pada subset | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Layerwise LR decay | optimization/amp_grad_accum | snippet:optimization/amp_grad_accum | Manual loop/LLRD kompleks bukan wajib; fp16/accumulation jangan menurunkan learning rate tanpa hitung ulang langkah. |
| Label smoothing | modeling/transformer_trainer | snippet:modeling/transformer_trainer | 84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit. |
| Auto-retry OOM | optimization/oom_fallback | snippet:optimization/oom_fallback | Retry model training di state yang sudah setengah berubah dapat mengulang langkah; simpan checkpoint, kecilkan batch eksplisit. |
| Inferensi terurut panjang | optimization/encode_cached | snippet:optimization/encode_cached | Memmap perlu metadata jumlah/urutan/model/prefix; jangan menerima cache hanya karena file ada. |
| Tuning secukupnya | modeling/linear_models | snippet:modeling/linear_models | LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend. |
| Optuna | modeling/tree_sparse | snippet:modeling/tree_sparse | Tree tidak selalu lebih baik pada TF-IDF; objective/class count dan early stopping perlu schema yang benar. |
| Pilih k topik/klaster | evaluation/metrics_topic | snippet:evaluation/metrics_topic | NPMI kasus p=1/term tak muncul perlu guard; coherence/diversity hanya diagnostik dan contoh manusia tetap wajib. |
| Seed dan config tunggal | optimization/seed_everything | snippet:optimization/seed_everything | Seed bukan jaminan bitwise deterministic semua CUDA; workers/algoritme dan split tetap dicatat. |
| Simpan artefak tiap run | optimization/save_artifacts | snippet:optimization/save_artifacts | Nama artifact/order label/versi harus ada; penyimpanan joblib singkat cukup untuk baseline. |
| Satu perubahan per run | optimization/save_artifacts | snippet:optimization/save_artifacts | Nama artifact/order label/versi harus ada; penyimpanan joblib singkat cukup untuk baseline. |
| Cek lingkungan | optimization/gpu_check | snippet:optimization/gpu_check | Periksa perangkat saat setup; runtime CPU fallback terlihat, RTX 3070 tidak diasumsikan mendukung bf16. |
| Replika metrik panitia | evaluation/metrics_classification | snippet:evaluation/metrics_classification | Kode lama mengasumsikan label integer 0..K-1; kartu baru memakai label asli dan mapping m.classes_. |
| Paket metrik klasifikasi | evaluation/metrics_classification | snippet:evaluation/metrics_classification | Kode lama mengasumsikan label integer 0..K-1; kartu baru memakai label asli dan mapping m.classes_. |
| Metrik multilabel | evaluation/metrics_classification | snippet:evaluation/metrics_classification | Kode lama mengasumsikan label integer 0..K-1; kartu baru memakai label asli dan mapping m.classes_. |
| Metrik regresi | evaluation/metrics_regression | snippet:evaluation/metrics_regression | RMSE API terbaru, MAE median, korelasi konstan undefined; QWK perlu ordinal dan cutpoint development. |
| Korelasi skor | evaluation/metrics_regression | snippet:evaluation/metrics_regression | RMSE API terbaru, MAE median, korelasi konstan undefined; QWK perlu ordinal dan cutpoint development. |
| Metrik klaster | evaluation/metrics_clustering | snippet:evaluation/metrics_clustering | Hungarian accuracy hanya saat ada gold dan definisinya cocok; internal metric bukan accuracy. |
| Metrik topik | evaluation/metrics_topic | snippet:evaluation/metrics_topic | NPMI kasus p=1/term tak muncul perlu guard; coherence/diversity hanya diagnostik dan contoh manusia tetap wajib. |
| Metrik retrieval | evaluation/metrics_retrieval | snippet:evaluation/metrics_retrieval | Cutoff MAP/MRR lama bukan @K; kartu baru cutoff eksplisit, multi-positive dan graded gain dipisah. |
| Entity-level F1 | evaluation/metrics_ner | snippet:evaluation/metrics_ner | Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat. |
| EM dan F1 token | evaluation/metrics_qa | snippet:evaluation/metrics_qa | Multi-reference ambil skor terbaik; normalisasi evaluator Indonesia tidak otomatis sama dengan SQuAD Inggris. |
| ROUGE | evaluation/metrics_rouge | snippet:evaluation/metrics_rouge | ROUGE sederhana tetap referensi; tokenisasi/stemming tidak setara evaluator resmi dan fakta perlu diaudit. |
| Top-k dan perplexity | evaluation/metrics_classification | snippet:evaluation/metrics_classification | Kode lama mengasumsikan label integer 0..K-1; kartu baru memakai label asli dan mapping m.classes_. |
| Bootstrap noise floor | evaluation/noise_floor | snippet:evaluation/noise_floor | Bootstrap 2SE adalah ketidakpastian estimator, bukan label-noise floor; group bootstrap jika data berkelompok. |
| Banding berpasangan | evaluation/noise_floor | snippet:evaluation/noise_floor | Bootstrap 2SE adalah ketidakpastian estimator, bukan label-noise floor; group bootstrap jika data berkelompok. |
| Kurva belajar | evaluation/noise_floor | snippet:evaluation/noise_floor | Bootstrap 2SE adalah ketidakpastian estimator, bukan label-noise floor; group bootstrap jika data berkelompok. |
| Selisih lokal vs platform | validation/leakage_overlap | snippet:validation/leakage_overlap | Overlap normalized text bukan satu-satunya leakage; dua endpoint/user/near-duplicate memerlukan grup sendiri. |
| Confusion matrix ternormalisasi | evaluation/error_analysis | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| Laporan per label | evaluation/error_analysis | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| Baca 50 error acak | evaluation/error_analysis | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| Skor per segmen | evaluation/error_analysis | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| Residual regresi | evaluation/error_analysis | snippet:evaluation/error_analysis | Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut. |
| Baca isi klaster/topik | evaluation/metrics_topic | snippet:evaluation/metrics_topic | NPMI kasus p=1/term tak muncul perlu guard; coherence/diversity hanya diagnostik dan contoh manusia tetap wajib. |
| Error span | evaluation/metrics_ner | snippet:evaluation/metrics_ner | Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat. |
| Error jawaban | evaluation/metrics_qa | snippet:evaluation/metrics_qa | Multi-reference ambil skor terbaik; normalisasi evaluator Indonesia tidak otomatis sama dengan SQuAD Inggris. |
| Threshold biner dari kurva OOF | imbalance/threshold_tuning | snippet:imbalance/threshold_tuning | Smoothing kurva dan nested row CV tidak otomatis menghormati grup; kartu memakai subset tuning/check per grup. |
| Pengali per kelas | postprocessing/class_thresholds | snippet:postprocessing/class_thresholds | Tuning/check grup dipisah; threshold rare label fallback dan tidak memaksa minimal satu label aktif. |
| Threshold per label | postprocessing/class_thresholds | snippet:postprocessing/class_thresholds | Tuning/check grup dipisah; threshold rare label fallback dan tidak memaksa minimal satu label aktif. |
| Top-k label dan label minimum | postprocessing/class_thresholds | snippet:postprocessing/class_thresholds | Tuning/check grup dipisah; threshold rare label fallback dan tidak memaksa minimal satu label aktif. |
| Cut point ordinal | postprocessing/ordinal_cutpoints | snippet:postprocessing/ordinal_cutpoints | Ordinal cutpoint memerlukan check independen dan kelas terurut; jangan menamai regresi biasa sebagai QWK optimal. |
| Inverse transform dan clip | postprocessing/ordinal_cutpoints | snippet:postprocessing/ordinal_cutpoints | Ordinal cutpoint memerlukan check independen dan kelas terurut; jangan menamai regresi biasa sebagai QWK optimal. |
| Perbaikan urutan BIO | evaluation/metrics_ner | snippet:evaluation/metrics_ner | Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat. |
| Agregasi per query | evaluation/metrics_retrieval | snippet:evaluation/metrics_retrieval | Cutoff MAP/MRR lama bukan @K; kartu baru cutoff eksplisit, multi-positive dan graded gain dipisah. |
| Decode span terbaik | modeling/qa_transformer | snippet:modeling/qa_transformer | Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi. |
| Ambang tidak terjawab | modeling/qa_transformer | snippet:modeling/qa_transformer | Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi. |
| Batas panjang ringkasan | modeling/summarize_extractive | snippet:modeling/summarize_extractive | Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang. |
| Gabung klaster kecil | modeling/cluster_hdbscan | snippet:modeling/cluster_hdbscan | Noise -1 tidak wajib dipaksa KNN ke cluster; HDBSCAN butuh skala/dimensi dan audit outlier. |
| Assign sisa baris | modeling/topic_embedding | snippet:modeling/topic_embedding | Fit topik di sampel awal/unique train, transform sisanya; topik klaster tanpa gold tidak punya accuracy. |
| Nama dan penggabungan topik | modeling/keyword_extract | snippet:modeling/keyword_extract | Ambil sparse top terms tanpa densify corpus; MMR diversity bukan metrik relevansi gold. |
| Rata-rata probabilitas | postprocessing/prob_average | snippet:postprocessing/prob_average | Ensemble hanya pada baris dan label order sama; logits/margin tidak dicampur sebagai probability. |
| Rata-rata rank | postprocessing/rank_average | snippet:postprocessing/rank_average | Rank average untuk ranking, bukan probability/log-loss; kandidat/cutoff dan group query harus sama. |
| Rata-rata regresi | postprocessing/prob_average | snippet:postprocessing/prob_average | Ensemble hanya pada baris dan label order sama; logits/margin tidak dicampur sebagai probability. |
| Seed dan fold bagging | postprocessing/prob_average | snippet:postprocessing/prob_average | Ensemble hanya pada baris dan label order sama; logits/margin tidak dicampur sebagai probability. |
| Stacking | postprocessing/stacking | snippet:postprocessing/stacking | OOF meta row CV tidak otomatis jujur per grup; nested base predictions perlu untuk estimasi ketat. Blend sederhana dulu. |
| Voting span atau entitas | evaluation/metrics_ner | snippet:evaluation/metrics_ner | Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat. |
| Temperature scaling | postprocessing/calibration | snippet:postprocessing/calibration | Temperature tidak mengubah argmax; perbaikan log-loss bukan otomatis peningkatan F1. |
| Platt dan isotonic | postprocessing/calibration | snippet:postprocessing/calibration | Temperature tidak mengubah argmax; perbaikan log-loss bukan otomatis peningkatan F1. |
| Aturan pola pasti | postprocessing/rule_overrides | snippet:postprocessing/rule_overrides | Lookup train atau regex hanya sesuai aturan dan audit train/gold; matching yang menghapus tanda baca tidak membuktikan identitas semantik. |
| Lookup duplikat dari train | postprocessing/rule_overrides | snippet:postprocessing/rule_overrides | Lookup train atau regex hanya sesuai aturan dan audit train/gold; matching yang menghapus tanda baca tidak membuktikan identitas semantik. |
| Default baris kosong | postprocessing/rule_overrides | snippet:postprocessing/rule_overrides | Lookup train atau regex hanya sesuai aturan dan audit train/gold; matching yang menghapus tanda baca tidak membuktikan identitas semantik. |
| Head + tail TTA | postprocessing/tta | snippet:postprocessing/tta | Head/tail/chunks bukan augmentasi valid semua task; QA offset dan NLI direction tidak boleh dirusak. |
| Kandidat A dan B | optimization/save_artifacts | snippet:optimization/save_artifacts | Nama artifact/order label/versi harus ada; penyimpanan joblib singkat cukup untuk baseline. |
| Validator submission | postprocessing/label_mapping | snippet:postprocessing/label_mapping | ID dibaca string agar nol depan utuh; format sample_submission dan label order wajib cocok. |
| Decode label | postprocessing/label_mapping | snippet:postprocessing/label_mapping | ID dibaca string agar nol depan utuh; format sample_submission dan label order wajib cocok. |
| Submit lebih awal | out |  | Contoh API ilustratif, bukan sel mandiri. Gunakan input dan langkah eksplisit kartu terkait. |
| Slide selagi lomba berjalan | output/slides | snippet:output/slides | PPTX optional output; tugas cheatsheet tidak perlu mengetik helper slide sebelum jawaban/data/model benar. |
| Gambar untuk slide | output/plots | snippet:output/plots | Grafik sederhana biru/kuning; bukan donut/dekorasi wajib dan angka EDA tetap tersedia. |
| Ceritakan dedup dan validasi | out |  | Contoh API ilustratif, bukan sel mandiri. Gunakan input dan langkah eksplisit kartu terkait. |
| Pembeda slide MBG | out |  | Contoh API ilustratif, bukan sel mandiri. Gunakan input dan langkah eksplisit kartu terkait. |
| Tiga rekomendasi berpemilik | output/slides | snippet:output/slides | PPTX optional output; tugas cheatsheet tidak perlu mengetik helper slide sebelum jawaban/data/model benar. |
| Augmentasi teks ringan | modeling/augment_text | snippet:modeling/augment_text | Word swap/dropout tidak menjamin label terjaga, terutama negasi/angka/NLI; hanya train dan validasi dampaknya. |
| Negatif hasil mutasi | feature_engineering/pair_features | snippet:feature_engineering/pair_features | 52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar. |
| Fine-tune seq2seq | modeling/seq2seq | snippet:modeling/seq2seq | Base model bukan translator/summarizer siap pakai; prefix dan language target bergantung model yang benar. |
| Kamus per kata | preprocessing/slang_id | snippet:preprocessing/slang_id | Kamus formal-domain spesifik, audit train; mapping berbatas kata, tidak substring replacement. |
| CER, WER, BLEU | evaluation/metrics_seq2seq | snippet:evaluation/metrics_seq2seq | CER/WER dapat >1; BLEU smoothing/tokenisasi buatan sendiri tidak boleh diberi label skor resmi. |
| Akurasi token | evaluation/metrics_ner | snippet:evaluation/metrics_ner | Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat. |
