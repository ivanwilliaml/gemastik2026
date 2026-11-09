# title: Pasangan teks: NLI, duplikat, konsistensi
# for: pair dupq consistency
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
A, B, TARGET = "text_a", "text_b", "label"
USE_BERT = False
BERT_MODEL, MAX_LEN, BATCH = MODELS / "indobertweet", 128, 32

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: eda | EDA
# note: Dua sisi, overlap, dan teks berulang.
# uses: eda/eda_report, validation/near_dup_groups
report = eda_report(train.assign(_all=train[A].astype(str) + " " + train[B].astype(str)), None, "_all", TARGET)
print("teks unik", train[A].nunique(), train[B].nunique(), "| grup", len(set(pair_groups(train[A], train[B]))))

# %% stage: clean | Cleaning data
# note: Isi kosong. Pasangan sama dibuang.
# uses: 
for d in (train, test):
    d[A], d[B] = d[A].fillna("").astype(str), d[B].fillna("").astype(str)
train = train.drop_duplicates([A, B]).reset_index(drop=True)
print(train.shape)

# %% stage: feat | Fitur relasi
# note: Overlap, Jaccard, kosinus, fuzzy, konflik angka, entitas, negasi.
# uses: feature_engineering/pair_features
X, vec = pair_features(train[A].tolist(), train[B].tolist())
Xt, _ = pair_features(test[A].tolist(), test[B].tolist(), vectorizer=vec)
print(X.describe().loc[["mean", "std"]].round(3).to_string())

# %% stage: split | Validasi
# note: Komponen graf: teks sama tidak melintasi fold.
# uses: postprocessing/label_mapping, validation/make_folds
y, classes = encode_labels(train[TARGET])
folds = make_folds(y, pair_groups(train[A], train[B]), kind="group", seed=SEED)

# %% stage: model | Model
# note: LightGBM dengan early stopping per fold.
# uses: modeling/tree_sparse, evaluation/metrics_classification, optimization/save_artifacts
oof, pred = np.zeros((len(y), len(classes))), np.zeros((len(test), len(classes)))
for tr, va in folds:
    m = fit_lgbm_es(make_lgbm(1000), X.iloc[tr], y[tr], X.iloc[va], y[va])
    oof[va] = m.predict_proba(X.iloc[va])
    pred += m.predict_proba(Xt) / len(folds)
print(cls_report(y, oof))
save_run("pair_lgbm", oof, pred, cls_report(y, oof)["macro_f1"])

# %% stage: bert | Cross-encoder
# note: Lewati bila USE_BERT=False.
# uses: modeling/pair_cross_encoder
bert_test = None
if USE_BERT:
    tr, va = folds[0]
    a, b = train[A].tolist(), train[B].tolist()
    m, tok = train_pair(BERT_MODEL, [a[i] for i in tr], [b[i] for i in tr], y[tr], [a[i] for i in va], [b[i] for i in va], y[va], num_labels=len(classes), max_len=MAX_LEN, batch=BATCH)
    print("cross-encoder", cls_report(y[va], predict_pair(m, tok, [a[i] for i in va], [b[i] for i in va], max_len=MAX_LEN)))
    bert_test = predict_pair(m, tok, test[A].tolist(), test[B].tolist(), max_len=MAX_LEN)
else:
    print("USE_BERT=False")

# %% stage: eval | Evaluasi & error analysis
# note: Kesalahan paling yakin.
# uses: evaluation/error_analysis
print(worst_errors((train[A] + " || " + train[B]).tolist(), y, oof, 10, classes).to_string())

# %% stage: post | Post-processing
# note: Blend dan threshold bila biner.
# uses: postprocessing/prob_average, imbalance/threshold_tuning
if bert_test is not None:
    pred = blend([pred, bert_test])
if len(classes) == 2:
    print(nested_threshold_score(y, oof[:, 1]))
    final = (pred[:, 1] >= best_threshold(y, oof[:, 1])).astype(int)
else:
    final = pred.argmax(1)

# %% stage: out | Submission
# note: Validasi format.
# uses: 
sub = write_submission(SAMPLE, classes[final], allowed=classes)
