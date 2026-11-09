# title: Multilabel
# for: multilabel
# %% stage: setup | Setup & konfigurasi
# note: Satu-satunya sel yang wajib diubah. Jalankan paling awal.
# uses: optimization/gpu_check, optimization/seed_everything
import re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
MODELS = Path(os.environ.get("COMP_ROOT", r"C:\gemastik")) / "models"
SEED = 42
seed_everything(SEED)
env_report(MODELS)

TRAIN, TEST, SAMPLE = "train.csv", "test.csv", "sample_submission.csv"
TEXT, LABELS = "text", ["l0", "l1", "l2"]        # kolom 0/1 per label
PREPROCESS, VEC, MODEL = "tfidf", "tfidf_union", "ovr"   # MODEL: ovr atau chain
USE_BERT = False
BERT_MODEL, MAX_LEN, BATCH = MODELS / "indobertweet", 128, 32

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: eda | EDA
# note: Termasuk label per baris dan label jarang.
# uses: eda/eda_report, eda/label_distribution
report = eda_report(train, test, TEXT, LABELS)
label_stats, cooc = multilabel_table(train[LABELS].to_numpy(), LABELS)
print(label_stats.to_string(), "\n", cooc.to_string())

# %% stage: clean | Cleaning data
# note: Dedup tanpa mengubah label.
# uses: noisy_data/clean_all
train = clean_dataset(train, TEXT, conflict=None)
test = clean_test(test, TEXT)

# %% stage: prep | Preprocessing
# note: Ganti preset di Setup.
# uses: preprocessing/preprocess_all
texts, test_texts = preprocess_corpus(train[TEXT], PREPROCESS), preprocess_corpus(test[TEXT], PREPROCESS)

# %% stage: feat | Fitur
# note: Nama vectorizer di Setup.
# uses: feature_engineering/make_vectorizer
print(make_vectorizer(VEC).fit_transform(texts[:5000]).shape)

# %% stage: split | Validasi
# note: Iterative stratification menjaga label jarang.
# uses: validation/make_folds
Y = train[LABELS].to_numpy()
folds = make_folds(Y, kind="multilabel", seed=SEED)
check_folds(texts, folds)

# %% stage: model | Model
# note: Satu model per label.
# uses: modeling/make_model, evaluation/metrics_classification, optimization/save_artifacts
oof, pred = oof_predict(lambda: make_vectorizer(VEC), lambda: make_model(MODEL), texts, Y, folds, test_texts)
print("threshold 0.5", multilabel_report(Y, oof >= .5))
save_run(MODEL, oof, pred, multilabel_report(Y, oof >= .5)["macro_f1"])

# %% stage: bert | Transformer
# note: Lewati bila USE_BERT=False.
# uses: modeling/transformer_multilabel
bert_valid, bert_test = None, None
if USE_BERT:
    tr, va = folds[0]
    m, tok = train_multilabel(BERT_MODEL, [texts[i] for i in tr], Y[tr], [texts[i] for i in va], Y[va], max_len=MAX_LEN, batch=BATCH)
    bert_valid, bert_test = (va, predict_multilabel(m, tok, [texts[i] for i in va], max_len=MAX_LEN)), predict_multilabel(m, tok, test_texts, max_len=MAX_LEN)
    print("BERT", multilabel_report(Y[va], bert_valid[1] >= .5))
else:
    print("USE_BERT=False")

# %% stage: eval | Evaluasi & error analysis
# note: Skor per label: label jarang biasanya sumber loss.
# uses: evaluation/error_analysis
for j, l in enumerate(LABELS):
    print(l, per_label(Y[:, j], (oof[:, j] >= .5).astype(int)).loc["macro avg", "f1-score"])

# %% stage: post | Post-processing
# note: Threshold per label dari OOF.
# uses: postprocessing/class_thresholds, postprocessing/prob_average
t = fit_label_thresholds(Y, oof)
print("threshold", dict(zip(LABELS, t.round(2))), multilabel_report(Y, apply_thresholds(oof, t)))
if bert_test is not None:
    pred = blend([pred, bert_test])
final = apply_thresholds(pred, t)

# %% stage: out | Submission
# note: Isi kolom label sesuai contoh.
# uses: postprocessing/label_mapping
sub = pd.read_csv(SAMPLE)
sub[LABELS] = final
sub.to_csv("submission.csv", index=False)
print(sub.shape, sub[LABELS].mean().round(3).to_dict())
