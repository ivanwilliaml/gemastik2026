# title: Regresi dan ordinal
# for: regression
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
TEXT, TARGET = "text", "label"
PREPROCESS = "tfidf"
EXPERIMENTS = [("tfidf_word", "ridge"), ("lsa", "lgbm_reg")]   # target berekor: ridge_log. MAE: lgbm_l1
ORDINAL = True                                   # True bila skor diskret dan metrik QWK
USE_BERT = False
BERT_MODEL, MAX_LEN, BATCH = MODELS / "indobertweet", 128, 32

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: eda | EDA
# note: Skew dan target diskret.
# uses: eda/eda_report, eda/label_distribution
report = eda_report(train, test, TEXT, TARGET)
print(target_table(train[TARGET]))

# %% stage: clean | Cleaning data
# note: Dedup: nilai target dirata-rata tidak, baris pertama dipertahankan.
# uses: noisy_data/clean_all
train, test = clean_dataset(train, TEXT, conflict=None), clean_test(test, TEXT)

# %% stage: prep | Preprocessing
# note: Ganti preset di Setup.
# uses: preprocessing/preprocess_all
texts, test_texts = preprocess_corpus(train[TEXT], PREPROCESS), preprocess_corpus(test[TEXT], PREPROCESS)

# %% stage: feat | Fitur
# note: Nama vectorizer di EXPERIMENTS.
# uses: feature_engineering/make_vectorizer
print(make_vectorizer("tfidf_word").fit_transform(texts[:5000]).shape)

# %% stage: split | Validasi
# note: Target di-bin agar tiap fold mewakili rentang.
# uses: validation/make_folds
y = train[TARGET].to_numpy(float)
folds = make_folds(y, kind="regression", seed=SEED)
check_folds(texts, folds)

# %% stage: model | Model
# note: Bandingkan dengan prediksi rata-rata.
# uses: modeling/make_model, evaluation/metrics_regression, optimization/save_artifacts
results = {}
print("baseline", reg_report(y, np.r_[np.full(len(y) - 1, y.mean()), y.mean() + 1e-6]))
for v, m in EXPERIMENTS:
    o, p = oof_predict(lambda: make_vectorizer(v, **(dict(k=64) if v == "lsa" else {})), lambda: make_model(m), texts, y, folds, test_texts)
    results[f"{v}+{m}"] = (o.ravel(), p.ravel())
    print(f"{v}+{m}", reg_report(y, o.ravel()))
    save_run(f"{v}+{m}", o, p, reg_report(y, o.ravel())["rmse"])
best = min(results, key=lambda k: reg_report(y, results[k][0])["rmse"])
oof, pred = results[best]
print("terbaik", best)

# %% stage: bert | Transformer
# note: Lewati bila USE_BERT=False.
# uses: modeling/transformer_regression
bert_test = None
if USE_BERT:
    tr, va = folds[0]
    m, tok, sc = train_regression(BERT_MODEL, [texts[i] for i in tr], y[tr], [texts[i] for i in va], y[va], max_len=MAX_LEN, batch=BATCH)
    print("BERT", reg_report(y[va], predict_regression(m, tok, [texts[i] for i in va], sc, max_len=MAX_LEN)))
    bert_test = predict_regression(m, tok, test_texts, sc, max_len=MAX_LEN)
else:
    print("USE_BERT=False")

# %% stage: eval | Evaluasi & error analysis
# note: Residual per rentang target dan noise floor.
# uses: evaluation/noise_floor, evaluation/error_analysis
from sklearn.metrics import root_mean_squared_error
print(noise_floor(y, oof, root_mean_squared_error, 200))
print(residual_table(y, oof).to_string())

# %% stage: post | Post-processing
# note: Blend, cut point ordinal, clip.
# uses: postprocessing/ordinal_cutpoints, postprocessing/prob_average
if bert_test is not None:
    pred = blend([pred, bert_test])
if ORDINAL:
    print("QWK bulat", round(qwk(y, np.rint(oof)), 4), "| cut nested", nested_cuts_score(y, oof))
    final = apply_cuts(pred, fit_cuts(y, oof), low=int(y.min()))
else:
    final = to_output(pred, lo=y.min(), hi=y.max())

# %% stage: out | Submission
# note: Kolom terakhir contoh submission.
# uses: postprocessing/label_mapping
sub = write_submission(SAMPLE, final)
