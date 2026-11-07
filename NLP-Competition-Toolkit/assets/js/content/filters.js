// Aturan filter task. Setiap item memiliki string `for` berisi token: id task, keluarga, atau alias.
window.TK = window.TK || {};
TK.families = {
  binary: ["binary", "spam", "fakenews"],
  multi: ["multiclass", "sentiment", "stance", "emotion", "intent", "langdetect"],
  mlabel: ["multilabel"],
  absa: ["absa"],
  reg: ["regression"],
  pair: ["pair", "dupq", "semsim", "consistency"],
  rank: ["ranking", "ir", "doc_sim"],
  topic: ["topic", "keyword"],
  cluster: ["doc_cluster", "emb_cluster", "dimred", "repr"],
  report: ["report", "trend", "mbg_extract"],
  anomaly: ["anomaly"],
  semi: ["pseudo", "selftrain", "confidence", "iterative", "weak"],
  ner: ["ner", "pos", "ate", "keyphrase"],
  qa: ["qa"],
  gen: ["summarization", "nextword", "translation", "normalization"]
};
TK.aliases = {
  // klasifikasi satu label per baris
  cls: ["binary", "multi", "semi"],
  // semua yang punya label untuk dilatih
  sup: ["binary", "multi", "semi", "mlabel", "absa", "reg", "pair", "ranking"],
  // keluaran berupa kelas diskret dan bisa dinilai F1
  disc: ["binary", "multi", "semi", "mlabel", "absa", "pair"],
  unsup: ["topic", "cluster", "report", "anomaly"],
  // pakai model klasifikasi/regresi dokumen
  doc: ["binary", "multi", "semi", "mlabel", "absa", "reg"],
  // teks penuh diumpankan ke transformer
  tf: ["binary", "multi", "semi", "mlabel", "absa", "reg", "pair", "ner", "qa", "ranking"],
  // retrieval dan similarity
  sim: ["pair", "rank"]
};
(function () {
  const cache = {};
  function expand(tok, out) {
    if (tok === "all") { out.all = true; return; }
    if (TK.aliases[tok]) TK.aliases[tok].forEach(t => expand(t, out));
    else if (TK.families[tok]) TK.families[tok].forEach(t => { out[t] = true; });
    else out[tok] = true;
  }
  // true bila item dengan string `forStr` berlaku untuk task terpilih
  TK.applies = function (forStr, task) {
    if (!task || !forStr) return true;
    if (!cache[forStr]) { const o = {}; forStr.split(/\s+/).forEach(t => expand(t, o)); cache[forStr] = o; }
    return !!(cache[forStr].all || cache[forStr][task]);
  };
  // Dipakai validator: token yang tidak dikenal
  TK.badTokens = function (forStr) {
    const ids = new Set(TK.tasks.map(t => t.id));
    return String(forStr).split(/\s+/).filter(t => t !== "all" && !TK.aliases[t] && !TK.families[t] && !ids.has(t));
  };
})();

// Slot yang tidak berlaku untuk semua task. Slot yang tidak terdaftar berlaku umum.
TK.slotFor = {
  "domain/label_source": "sup",
  "domain/temporal": "report sup",
  "domain/wilayah": "report ner sup",
  "domain/extract_structured": "report ner",
  "data_loading/load_conll": "ner",
  "eda/label_distribution": "sup ner",
  "eda/class_keywords": "disc",
  "preprocessing/branch_traditional": "topic cluster report anomaly",
  "feature_engineering/pair_features": "pair",
  "feature_engineering/svd_reduce": "sup unsup",
  "feature_engineering/word2vec_train": "sup unsup",
  "feature_engineering/fasttext_train": "sup unsup",
  "feature_engineering/doc2vec_train": "sup unsup",
  "validation/kfold_stratified": "disc reg",
  "validation/multilabel_split": "mlabel absa",
  "modeling/linear_models": "cls mlabel absa",
  "modeling/nb_complement": "cls",
  "modeling/tree_sparse": "doc pair",
  "modeling/random_forest_svd": "doc pair",
  "modeling/regression_linear": "reg semsim",
  "modeling/multilabel_ovr": "mlabel absa",
  "modeling/cnn_text": "doc",
  "modeling/rnn_text": "doc nextword ner",
  "modeling/attention_rnn": "doc",
  "modeling/transformer_trainer": "cls pair absa",
  "modeling/transformer_multilabel": "mlabel",
  "modeling/transformer_regression": "reg semsim",
  "modeling/pair_biencoder": "pair",
  "modeling/pair_cross_encoder": "pair ranking",
  "modeling/topic_nmf_lda": "topic report",
  "modeling/topic_embedding": "topic report",
  "modeling/cluster_kmeans": "cluster report",
  "modeling/cluster_hdbscan": "cluster report",
  "modeling/keyword_extract": "topic report",
  "modeling/summarize_extractive": "summarization",
  "modeling/seq2seq": "gen",
  "modeling/augment_text": "cls",
  "modeling/nn_train": "doc nextword ner",
  "evaluation/metrics_seq2seq": "translation normalization",
  "feature_engineering/nb_log_ratio": "binary sentiment",
  "feature_engineering/graph_features": "dupq",
  "modeling/ner_baseline": "ner",
  "modeling/ner_transformer": "ner",
  "modeling/retrieval_bm25": "rank qa",
  "modeling/retrieval_dense": "rank qa",
  "modeling/ranker_lgbm": "ranking",
  "modeling/lang_detect": "langdetect",
  "modeling/ngram_next_word": "nextword",
  "modeling/anomaly_text": "anomaly",
  "modeling/qa_extractive": "qa",
  "modeling/qa_transformer": "qa",
  "modeling/absa_pair_input": "absa",
  "modeling/absa_per_aspect": "absa",
  "modeling/zero_shot": "sup report weak",
  "modeling/setfit": "cls",
  "modeling/head_on_embed": "doc pair",
  "modeling/hashing_sgd": "doc",
  "evaluation/metrics_classification": "disc",
  "evaluation/metrics_regression": "reg semsim",
  "evaluation/metrics_clustering": "cluster report",
  "evaluation/metrics_topic": "topic report",
  "evaluation/metrics_retrieval": "rank keyword",
  "evaluation/metrics_ner": "ner",
  "evaluation/metrics_qa": "qa",
  "evaluation/metrics_rouge": "summarization",
  "evaluation/error_analysis": "sup ner qa",
  "evaluation/noise_floor": "sup ner",
  "imbalance/class_weights": "disc",
  "imbalance/threshold_tuning": "binary pair",
  "imbalance/resampling": "cls absa",
  "imbalance/smote_svd": "cls",
  "imbalance/focal_loss": "disc",
  "distribution_shift/adversarial_validation": "sup ner",
  "distribution_shift/importance_weights": "doc pair",
  "noisy_data/conflicting_labels": "sup",
  "noisy_data/label_noise_oof": "disc",
  "semi_supervised/pseudo_label": "cls",
  "semi_supervised/self_training": "cls",
  "semi_supervised/confidence_filter": "cls",
  "semi_supervised/iterative_pseudo": "cls",
  "semi_supervised/weak_supervision": "cls report",
  "postprocessing/prob_average": "disc",
  "postprocessing/rank_average": "binary ranking pair",
  "postprocessing/stacking": "disc reg",
  "postprocessing/class_thresholds": "multi mlabel absa semi",
  "postprocessing/calibration": "cls mlabel pair",
  "postprocessing/rule_overrides": "sup",
  "postprocessing/label_mapping": "disc",
  "postprocessing/tta": "doc",
  "postprocessing/ordinal_cutpoints": "reg"
};

TK.strategyFor = {
  imbalance: "disc ner", shift: "sup ner qa", "noisy-labels": "sup", missing: "all", duplicates: "all",
  length: "all", oov: "all", multilang: "all", "no-labels": "unsup semi", "train-test-diff": "sup ner qa",
  fit: "sup ner qa", vram: "all", time: "all", "no-pretrained": "all"
};
TK.diagFor = { imb: "disc", shift: "sup ner qa", noise: "sup", valid: "sup ner qa", other: "all" };

// Tahap pipeline ke tahap katalog metode, dan fungsi utuh tiap tahap.
TK.stageMap = { setup: "train", bert: "model", label: "eda", insight: "out", tag: "feat", extract: "feat", geo: "eval", ner: "model", trend: "eval" };
TK.stageMain = { label: "domain/label_source", insight: "domain/mbg_lexicon", tag: "domain/mbg_lexicon", extract: "domain/extract_structured", geo: "domain/wilayah", trend: "domain/temporal", load: "data_loading/load_table", eda: "eda/eda_report", clean: "noisy_data/clean_all", prep: "preprocessing/preprocess_all", feat: "feature_engineering/make_vectorizer", split: "validation/make_folds", model: "modeling/make_model", train: "modeling/transformer_trainer", eval: "evaluation/metrics_classification", post: "postprocessing/class_thresholds", out: "postprocessing/label_mapping" };
