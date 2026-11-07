# title: Skor kemiripan
# for: semsim
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

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: feat | Fitur relasi
# note: Kosinus saja sudah menjadi baseline.
# uses: feature_engineering/pair_features
a, b, ta, tb = train[A].fillna("").tolist(), train[B].fillna("").tolist(), test[A].fillna("").tolist(), test[B].fillna("").tolist()
X, vec = pair_features(a, b)
Xt, _ = pair_features(ta, tb, vectorizer=vec)

# %% stage: split | Validasi
# note: Grup teks.
# uses: validation/near_dup_groups, validation/make_folds
y = train[TARGET].to_numpy(float)
folds = make_folds(y, pair_groups(a, b), kind="group", seed=SEED)

# %% stage: model | Model
# note: LightGBM pada fitur pasangan.
# uses: modeling/make_model, evaluation/metrics_regression
oof, pred = np.zeros(len(y)), np.zeros(len(ta))
for tr, va in folds:
    m = make_model("lgbm_reg", n_estimators=300).fit(X.iloc[tr], y[tr])
    oof[va] = m.predict(X.iloc[va])
    pred += m.predict(Xt) / len(folds)
print("kosinus saja", reg_report(y, X.cos_tfidf), "\nmodel", reg_report(y, oof))

# %% stage: out | Submission
# note: Clip ke rentang target.
# uses: postprocessing/label_mapping
sub = write_submission(SAMPLE, np.clip(pred, y.min(), y.max()))
