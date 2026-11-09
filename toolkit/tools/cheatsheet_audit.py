"""Disposition per file: jangan menyamakan static audit dengan uji model di data baru."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

# File | kartu pengganti yang diketik | alasan/kondisi hasil review.
REVIEWS='''
data_loading/detect_columns|load,profile|Tebakan ID/target dari kardinalitas tidak boleh menentukan schema otomatis; tetapkan kolom dari soal.
data_loading/duckdb_query|formats|SQL opsional file besar, identifier kolom perlu escaping; bukan helper wajib pembacaan kecil.
data_loading/load_conll|ner_load|Parser CoNLL tetap referensi format khusus; jangan hilangkan batas dokumen atau mengubah offset.
data_loading/load_table|load,formats|Cache parquet bisa kedaluwarsa dan on_bad_lines bisa kehilangan baris; pembacaan eksplisit lebih mudah diaudit.
data_loading/reduce_memory|profile|Downcast dapat mengubah presisi; category tidak wajib untuk teks, ukur RAM sebelum menyalin DataFrame.
distribution_shift/adversarial_validation|time_split,user_groups|Vectorizer lama fit gabungan sebelum CV domain; diagnostik transduktif bukan estimasi generalisasi induktif.
distribution_shift/domain_adaptation|tfidf|Fit gabungan train-test bersifat transduktif dan bersyarat aturan; CORAL dense bukan baseline yang wajib diketik.
distribution_shift/feature_shift|length,temporal|PSI konstan/kosong dan smoothing centered perlu perhatian; waktu online tidak boleh memakai periode masa depan.
distribution_shift/importance_weights|lr|Density ratio perlu probabilitas domain out-of-fold terkalibrasi dan ESS; tidak default pada data tak dikenal.
distribution_shift/length_shift|length|KS p-value dipengaruhi jumlah data; laporkan besar shift/persentil dan jangan menganggap p kecil bukti dampak model.
distribution_shift/vocab_shift|vocab|Log-ratio vocabulary hanya deskriptif; jangan otomatis menghapus kata khas test atau mengoptimasi dari label test.
domain/extract_structured|extract|Aturan uang/tanggal/NER domain panjang; output perlu bukti teks dan audit format lokal, bukan dianggap fakta.
domain/label_source|target|Agreement tinggi tidak membuktikan asal label; klaim lama dilunakkan dan heuristik bukan gold.
domain/language_mix|language|Penanda kata daerah adalah heuristik coverage kecil; teks campuran/pendek perlu audit manusia.
domain/mbg_lexicon|weak,extract|Kamus MBG kondisional, label isu bukan sentimen; abstain saat konflik dan audit precision pada gold independen.
domain/temporal|time_split,temporal|Pisahkan volume dan proporsi; topic ID harus beku, dukungan kecil dan perubahan rezim dilaporkan.
domain/wilayah|extract|Pencocokan/fuzzy wilayah bukan lokasi kejadian; kamus dan peta tetap referensi, bukan 152 baris wajib.
eda/class_keywords|target,evaluate|Kata pembeda dapat artefak; classifier kata kunci tidak membuktikan asal anotasi label.
eda/eda_report|profile,length,vocab,noise|101 baris helper diganti empat sensus literal; penyebut, sampel, hapax dan noise terlihat langsung.
eda/label_distribution|target,ml_eda|Unknown multilabel perlu penyebut diketahui; biner, multiclass, numeric dan ABSA tidak ditangani dengan rumus sama.
eda/length_stats|length,token_length|Persentil kata bukan token model; cutoff BERT dari tokenizer, bukan rata-rata karakter.
eda/quick_profile|profile|Missing/blank/pseudo-null dibedakan; tidak otomatis membuang kata NA atau memodifikasi test.
eda/vocab_stats|vocab|Hapax frekuensi token berbeda dari document-frequency; batas sampel ditulis agar pembaca tidak mengira sensus penuh.
evaluation/error_analysis|evaluate,ml_eval|Urutan label dan support harus tetap, jangan hanya membaca head data yang terurut.
evaluation/metrics_classification|evaluate,ml_eval|Kode lama mengasumsikan label integer 0..K-1; kartu baru memakai label asli dan mapping m.classes_.
evaluation/metrics_clustering|cluster_eval|Hungarian accuracy hanya saat ada gold dan definisinya cocok; internal metric bukan accuracy.
evaluation/metrics_ner|ner_eval,pos_eval|Exact BIO entity span berbeda dari token accuracy/POS; invalid BIO dan truncation harus terlihat.
evaluation/metrics_qa|qa_eval|Multi-reference ambil skor terbaik; normalisasi evaluator Indonesia tidak otomatis sama dengan SQuAD Inggris.
evaluation/metrics_regression|ridge,sts_model|RMSE API terbaru, MAE median, korelasi konstan undefined; QWK perlu ordinal dan cutpoint development.
evaluation/metrics_retrieval|retrieval_eval,rank_eval|Cutoff MAP/MRR lama bukan @K; kartu baru cutoff eksplisit, multi-positive dan graded gain dipisah.
evaluation/metrics_rouge|summary|ROUGE sederhana tetap referensi; tokenisasi/stemming tidak setara evaluator resmi dan fakta perlu diaudit.
evaluation/metrics_seq2seq|gen_metrics|CER/WER dapat >1; BLEU smoothing/tokenisasi buatan sendiri tidak boleh diberi label skor resmi.
evaluation/metrics_topic|topic_eval|NPMI kasus p=1/term tak muncul perlu guard; coherence/diversity hanya diagnostik dan contoh manusia tetap wajib.
evaluation/noise_floor|evaluate|Bootstrap 2SE adalah ketidakpastian estimator, bukan label-noise floor; group bootstrap jika data berkelompok.
feature_engineering/count_bow|tfidf|Counts tetap alternatif NB/LDA; batas vocabulary dan vocabulary train-only harus eksplisit.
feature_engineering/doc2vec_train|word2vec|Doc2Vec inference stokastik dan corpus kecil tidak stabil; dipindah dari jalur ketik inti ke referensi.
feature_engineering/fasttext_train|word2vec|FastText dapat membantu subword/OOV tetapi bukan wajib; train embedding pada train fold saja.
feature_engineering/graph_features|pair_split|Graph yang dibangun dari test/pasangan yang dinilai transduktif; gunakan graph untuk grouping, bukan leakage fitur tersembunyi.
feature_engineering/hashing|tfidf|Hashing berguna skala besar, collision dan sign memengaruhi NB; TF-IDF lebih jelas untuk baseline dan audit kata.
feature_engineering/make_vectorizer|tfidf,char,union,embed|Factory 94 baris dan cache global terlalu berat untuk ditulis; tiap representasi menjadi sel eksplisit.
feature_engineering/meta_features|length,noise|Fitur volume seluruh hari dapat melihat masa depan; jangan membocorkan sumber/test lewat meta feature tanpa audit shift.
feature_engineering/nb_log_ratio|lr|NB-ratio lama hardcode y==1; biner positif harus didefinisikan, tidak default untuk multiclass.
feature_engineering/pair_features|pair_features,pair_eda|52 baris fitur heuristic diringkas relasi sparse; mutasi angka/acak kata tidak menjamin negatif semantik benar.
feature_engineering/sbert_encode|embed|Cache sebelumnya tidak mengikat data/prefix/urutan; checksum ditambahkan, kartu pendek tidak memakai cache tersembunyi.
feature_engineering/svd_reduce|lsa|Batasi dim oleh ukuran train; UMAP/t-SNE untuk visual sampel bukan generalisasi dan hindari densify besar.
feature_engineering/tfidf_char|char|Default 500k fitur terlalu besar untuk baseline cepat; cap 60k dan min_df=1 untuk data kecil/emoji.
feature_engineering/tfidf_union|union|Default total 800k fitur bukan titik awal lomba; cap masing-masing 40k dan bandingkan baseline word dahulu.
feature_engineering/tfidf_word|tfidf|min_df=2/max_df=.95 bisa empty vocabulary pada data kecil; default pendek min_df=1 dengan peringatan emoji/blank.
feature_engineering/transformer_embed|embed|Mask padding benar wajib; base BERT mean pool tidak otomatis semantic embedding berkualitas, pilih E5 dengan prefix.
feature_engineering/word2vec_train|word2vec|32 baris sklearn wrapper dipangkas; OOV nol terlihat, workers=1 untuk reproducibility yang lebih baik.
imbalance/class_weights|lr,ml_model|Bobot dihitung dari train saja dan urut label konsisten; balanced bukan jaminan menang atau kalibrasi baik.
imbalance/focal_loss|bert_train|Focal loss opsional setelah baseline; bobot dan gamma belum tentu cocok dataset lain, tidak wajib diketik.
imbalance/resampling|lr|Resampling hanya train; koreksi prior butuh prior target yang sah, tidak boleh ditebak dari label test.
imbalance/smote_svd|lr,lsa|SMOTE sparse mentah dan kelas minor <=k bermasalah; bobot kelas lebih sederhana sebelum interpolasi embedding.
imbalance/threshold_tuning|threshold|Smoothing kurva dan nested row CV tidak otomatis menghormati grup; kartu memakai subset tuning/check per grup.
modeling/absa_pair_input|absa_clean|Long format harus dibuat setelah split komentar; missing unknown berbeda dari aspek tidak ada.
modeling/absa_per_aspect|absa_model|Satu model per aspek, label kategori dan missing dimasking; kelas train konstan dilaporkan.
modeling/anomaly_text|anomaly|Anomali bukan label noise; threshold/top-k perlu audit gold dan jangan otomatis menghapus titik langka.
modeling/attention_rnn|bert_train|Arsitektur neural custom membawa trainer/tokenizer tambahan; referensi lanjutan, bukan jalur ketik pertama.
modeling/augment_text|normalize|Word swap/dropout tidak menjamin label terjaga, terutama negasi/angka/NLI; hanya train dan validasi dampaknya.
modeling/cluster_hdbscan|cluster|Noise -1 tidak wajib dipaksa KNN ke cluster; HDBSCAN butuh skala/dimensi dan audit outlier.
modeling/cluster_kmeans|cluster,cluster_eval|Pencarian k dibatasi development dan silhouette sampel; consensus bukan bukti cluster semantik benar.
modeling/cnn_text|lr|CNN custom memerlukan kosakata/trainer/padding; tidak worth mengetik sebelum baseline linear selesai.
modeling/hashing_sgd|tfidf|Untuk data sangat besar gunakan streaming SGDC loss log_loss; urutan kelas partial_fit harus konsisten.
modeling/head_on_embed|embed_head|Embedding beku boleh dicache dengan mapping baris jelas; pretrained sentiment head tidak otomatis cocok label domain.
modeling/keyword_extract|keywords|Ambil sparse top terms tanpa densify corpus; MMR diversity bukan metrik relevansi gold.
modeling/lang_detect|language|Language marker dan pretrained language-ID bukan gold, code-mixing/teks pendek belum tentu terdeteksi benar.
modeling/linear_models|lr,svm|LR/SVM pendek dipisah: margin SVM bukan probability, jangan menggunakan sigmoid margin untuk log-loss/blend.
modeling/make_model|lr,svm,ridge|Factory 118 baris dan arsitektur raw-text tidak membantu ketik ulang; impor estimator langsung.
modeling/multilabel_ovr|ml_model|OneVsRest biasa tidak memasking partial label; head per label train-known diperlukan.
modeling/nb_complement|lr|NB hanya fitur nonnegatif; boleh kandidat counts/TF-IDF, bukan SVD negatif atau semua representasi.
modeling/ner_baseline|ner_crf|CRF fitur token dibuka langsung; gazetteer dan majority token bukan pengganti exact-span evaluation.
modeling/ner_transformer|ner_load,ner_eval|Truncation lama mengisi O pada token hilang; audit coverage/stride, jangan mengklaim full tagging setelah crop.
modeling/ngram_next_word|nextword|Split dokumen sebelum membuat prefix; input evaluasi tidak boleh mengandung kata jawaban.
modeling/nn_train|bert_train|Custom training loop 49 baris ditambah tokenizer/arsitektur terlalu banyak; advanced reference saja.
modeling/pair_biencoder|embed,retrieval_dense|Cosine membutuhkan embedding/prefix sesuai task; similarity bukan entailment probability.
modeling/pair_cross_encoder|pair_clean,pair_features|Pair tokenizer dua input lebih tepat daripada teks SEP manual; simetri tidak diterapkan pada NLI.
modeling/qa_extractive|qa_baseline|Sentence retrieval baseline bukan span answer akurat; jawaban-tail regex tidak menjamin jawaban benar.
modeling/qa_transformer|qa_reader|Span decode butuh stride, constraint panjang dan null score; pipeline reader lebih ringkas, no-answer perlu validasi.
modeling/random_forest_svd|lsa,lr|RF di sparse vocabulary besar bukan default cepat; SVD dibatasi dan fit train, jangan densify tanpa ukur RAM.
modeling/ranker_lgbm|rank_model|Fungsi lama np.asarray(X) tidak cocok sparse; kartu baru menjaga sparse dan group size sesuai urutan qid.
modeling/regression_linear|ridge|Regresi RMSE/MAE/RMSLE/ordinal berbeda; transform log tidak otomatis tepat dan nonnegative harus diperiksa.
modeling/retrieval_bm25|bm25|Tokenisasi dan ranking ID dijelaskan; top-k per query menghindari query×corpus dense.
modeling/retrieval_dense|retrieval_dense,retrieval_eval|Chunk aggregation per doc dan self-match perlu perhatian; prefix/query-document berbeda.
modeling/rnn_text|lr|RNN dari nol memerlukan trainer/kosakata; simpan sebagai referensi, tidak tampil sebagai sel inti cheatsheet.
modeling/seq2seq|generation|Base model bukan translator/summarizer siap pakai; prefix dan language target bergantung model yang benar.
modeling/setfit|embed_head|Embedding+LR lebih sedikit kode; SetFit optional data sedikit, tidak otomatis lebih baik dari baseline.
modeling/summarize_extractive|summary|Urutan kalimat dipertahankan, audit panjang/fakta; TextRank tidak harus menjadi helper graf panjang.
modeling/topic_embedding|topic_model,cluster|Fit topik di sampel awal/unique train, transform sisanya; topik klaster tanpa gold tidak punya accuracy.
modeling/topic_nmf_lda|topic_model,lda|LDA memakai counts; NMF TF-IDF; k dibatasi rank data dan contoh topik harus diaudit.
modeling/transformer_multilabel|ml_model|Masked BCE ditambahkan untuk unknown; sigmoid independen, bukan softmax/argmax dan bukan label float otomatis cukup.
modeling/transformer_regression|ridge,sts_model|num_labels=1 dan float target; STS simetris berbeda dari kelas dan skala skor harus jelas.
modeling/transformer_trainer|bert_setup,bert_train|84 baris universal dipecah tiga sel; custom loss kwargs diperbaiki dan path offline eksplisit.
modeling/tree_sparse|lr,rank_model|Tree tidak selalu lebih baik pada TF-IDF; objective/class count dan early stopping perlu schema yang benar.
modeling/zero_shot|weak|NLI zero-shot memberi label lemah, bukan gold; multi score membandingkan entail/contradiction, neutral diabaikan.
noisy_data/clean_all|profile,clean,split|Default majority-label correction sebelum split dihapus; dedup sekarang mempertahankan label konflik dalam grup sama.
noisy_data/conflicting_labels|target|Konflik ditampilkan dan dikelompokkan, tidak otomatis diputuskan ke label mayoritas sebelum evaluasi.
noisy_data/duplicates|profile,split|Normalisasi regex yang membuang tanda baca dapat menyatukan negasi/angka berbeda; grouping minimal lebih konservatif.
noisy_data/label_noise_oof|evaluate|Cleanlab/low confidence hanya prioritas audit, bukan bukti salah anotasi atau alasan drop otomatis.
noisy_data/text_outliers|noise,anomaly|Noise regex deskriptif; emoji/non-ASCII/dialek tidak otomatis data buruk.
optimization/amp_grad_accum|bert_train|Manual loop/LLRD kompleks bukan wajib; fp16/accumulation jangan menurunkan learning rate tanpa hitung ulang langkah.
optimization/encode_cached|embed|Memmap perlu metadata jumlah/urutan/model/prefix; jangan menerima cache hanya karena file ada.
optimization/gpu_check|bert_setup|Periksa perangkat saat setup; runtime CPU fallback terlihat, RTX 3070 tidak diasumsikan mendukung bf16.
optimization/oom_fallback|bert_train|Retry model training di state yang sudah setengah berubah dapat mengulang langkah; simpan checkpoint, kecilkan batch eksplisit.
optimization/race_plan|lr|Gate/scale heuristic bukan benchmark PC panitia; tidak membuang char/fold diam-diam hanya karena jumlah baris.
optimization/save_artifacts|save|Nama artifact/order label/versi harus ada; penyimpanan joblib singkat cukup untuk baseline.
optimization/seed_everything|load|Seed bukan jaminan bitwise deterministic semua CUDA; workers/algoritme dan split tetap dicatat.
optimization/subsample_stratified|split|Sampel stratified tidak dapat menyelamatkan kelas cuma satu grup; jangan subsample lalu berbagi duplicate dengan validation.
optimization/throughput_timer|bert_train|Estimasi waktu dari batch pemanasan perlu margin; tidak menjanjikan semua pretrained selesai pada PC panitia.
output/plots|plot_bar|Grafik sederhana biru/kuning; bukan donut/dekorasi wajib dan angka EDA tetap tersedia.
output/slides|plot_bar|PPTX optional output; tugas cheatsheet tidak perlu mengetik helper slide sebelum jawaban/data/model benar.
postprocessing/calibration|calibrate|Temperature tidak mengubah argmax; perbaikan log-loss bukan otomatis peningkatan F1.
postprocessing/class_thresholds|class_bias,ml_threshold|Tuning/check grup dipisah; threshold rare label fallback dan tidak memaksa minimal satu label aktif.
postprocessing/label_mapping|final|ID dibaca string agar nol depan utuh; format sample_submission dan label order wajib cocok.
postprocessing/ordinal_cutpoints|ridge|Ordinal cutpoint memerlukan check independen dan kelas terurut; jangan menamai regresi biasa sebagai QWK optimal.
postprocessing/prob_average|ensemble|Ensemble hanya pada baris dan label order sama; logits/margin tidak dicampur sebagai probability.
postprocessing/rank_average|retrieval_eval|Rank average untuk ranking, bukan probability/log-loss; kandidat/cutoff dan group query harus sama.
postprocessing/rule_overrides|target,evaluate|Lookup train atau regex hanya sesuai aturan dan audit train/gold; matching yang menghapus tanda baca tidak membuktikan identitas semantik.
postprocessing/stacking|ensemble|OOF meta row CV tidak otomatis jujur per grup; nested base predictions perlu untuk estimasi ketat. Blend sederhana dulu.
postprocessing/tta|pair_clean|Head/tail/chunks bukan augmentasi valid semua task; QA offset dan NLI direction tidak boleh dirusak.
preprocessing/branch_advanced|normalize,tweet_prep|Repeat/number/HTML switches tidak default untuk semua data; spam/QA bisa kehilangan sinyal/offset.
preprocessing/branch_minimal|clean|NFC dan whitespace eksplisit; NFKC/casefold dapat mengubah entitas dan offset, tidak otomatis aman semua task.
preprocessing/branch_traditional|normalize|Stemming/stopwords bukan wajib classifier; negasi/daerah dijaga dan uji pada development.
preprocessing/branch_transformer|clean,tweet_prep|Native dan BERTweet dipisah; pasangan diberikan lewat tokenizer dua input, bukan SEP literal universal.
preprocessing/preprocess_all|clean,normalize|98 baris saklar diganti cleaning kecil + normalisasi pilihan; kode yang diketik terlihat seluruhnya.
preprocessing/slang_id|normalize|Kamus formal-domain spesifik, audit train; mapping berbatas kata, tidak substring replacement.
preprocessing/stopwords_id|clean|Negasi tetap dipertahankan; stopwords dipakai bila cocok interpretasi topik, bukan default classifier.
semi_supervised/confidence_filter|pseudo|Confidence tinggi belum tentu benar/terkalibrasi; cek coverage per kelas dan gold tidak ikut pool.
semi_supervised/iterative_pseudo|pseudo|Iterasi memilih best pada gold menjadikannya development; tetap perlu final gold independen untuk klaim.
semi_supervised/pseudo_label|pseudo|Fit ulang vocabulary+classifier tidak memasukkan validation; pool duplicate/penulis lintas gold diaudit.
semi_supervised/self_training|pseudo|Label sentinel -1 hanya untuk unlabeled task tunggal; model harus predict_proba dan gold validation tetap terpisah.
semi_supervised/weak_supervision|weak|Voting konflik lama memaksa argmax/tie; kartu baru abstain saat multi-hit/no-hit dan audit coverage/precision.
validation/kfold_group|split,user_groups|Unit independen ditetapkan dari data, bukan sekadar random group; jumlah grup dan kelas tiap fold dicek.
validation/kfold_stratified|split|Stratify tidak menghentikan leakage duplicate dan gagal pada rare class; gunakan grouping/time sesuai test.
validation/kfold_strat_group|split,user_groups|SGKF hanya mendekati stratifikasi di bawah batas grup; proporsi dan missing class tetap diperiksa.
validation/leakage_overlap|split,pair_split|Overlap normalized text bukan satu-satunya leakage; dua endpoint/user/near-duplicate memerlukan grup sendiri.
validation/make_folds|split,pair_split|Multilabel groups sebelumnya diabaikan, NaN cast bool aktif; diperbaiki. OOF margin palsu probability kini ditolak.
validation/multilabel_split|ml_clean,split|NaN dahulu menjadi True; kini positive/known profile. Split rows tanpa grup belum aman terhadap duplicate.
validation/near_dup_groups|pair_split,user_groups|Connected components bisa sangat besar; near-duplicate MinHash heuristic tidak membuktikan label sama.
validation/oof_predict|split,lr|OOF belum terisi kini NaN, kolom kelas disejajarkan; sigmoid/softmax margin tidak diterima sebagai probability.
validation/split_holdout|split|Stratified rows saja bocor pada teks/pengguna berulang; default grouped holdout dengan assert keterpisahan.
validation/split_time|time_split|Timestamp invalid/tie/timezone dan recurrence harus dilaporkan; jangan memecah tanggal sama demi ukuran exact.
'''


def audit(cards):
    policy={}
    for line in REVIEWS.strip().splitlines():
        key,targets,note=line.split('|',2)
        assert key not in policy,key
        policy[key]=(targets.split(','),note)
    rows=[]
    for path in sorted((ROOT/'snippets').rglob('*.py')):
        key=path.relative_to(ROOT/'snippets').with_suffix('').as_posix()
        assert key in policy,f'Belum ditinjau: {key}'
        targets,note=policy.pop(key)
        assert all(target in cards for target in targets),(key,targets)
        code=path.read_text(encoding='utf-8');tree=ast.parse(code)
        functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        rows.append({'id':key,'path':'snippets/'+key+'.py','lines':len(code.splitlines()),
            'longest_function':max([n.end_lineno-n.lineno+1 for n in functions],default=0),
            'sha256':hashlib.sha256(code.encode()).hexdigest(),'action':'Diganti pada cheatsheet; sumber detail tetap referensi',
            'replacements':targets,'note':note,'verification':'AST + audit kondisi pemakaian; bukan klaim model unggul pada data baru'})
    assert not policy,policy.keys()
    return rows
