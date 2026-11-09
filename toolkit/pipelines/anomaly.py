# title: Anomali teks
# for: anomaly
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

DATA, TEXT = "train.csv", "text"
POOL = ["zxcv qwer asdf lorem ipsum", "beli slot gacor promo klik sekarang", "aaaa aaaa aaaa aaaa"]   # untuk uji injeksi

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)

# %% stage: prep | Preprocessing
# note: Teks dipertahankan agar keanehan tetap terlihat.
# uses: preprocessing/preprocess_all
docs = preprocess_corpus(df[TEXT], "tfidf")

# %% stage: model | Skor anomali
# note: Gabungan IForest, kNN, dan rekonstruksi.
# uses: feature_engineering/make_vectorizer, modeling/anomaly_text
def scores_of(t):
    X = make_vectorizer("tfidf_word", min_df=1).fit_transform(t)
    Z = make_vectorizer("lsa", k=32).fit_transform(t)
    return rank_avg(iforest_score(Z), knn_score(Z, 10), recon_score(X, 32))

df["anomaly"] = scores_of(docs)

# %% stage: eval | Uji injeksi
# note: Seed uji berbeda dari tuning. Periksa korelasi dengan panjang.
# uses: 
print(inject_eval(docs, scores_of, POOL, seed=7))
print("korelasi panjang", round(df["anomaly"].corr(df[TEXT].fillna("").str.len()), 3))
print(df.sort_values("anomaly", ascending=False)[TEXT].head(10).to_string())

# %% stage: out | Output
# note: Skor per baris.
# uses: 
df.to_csv("submission.csv", index=False)
