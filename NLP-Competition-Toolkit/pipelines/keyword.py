# title: Kata kunci
# for: keyword
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

DATA, TEXT, K = "train.csv", "text", 5

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)

# %% stage: prep | Preprocessing
# note: Stopword dibuang untuk kata kunci.
# uses: preprocessing/preprocess_all
docs = preprocess_corpus(df[TEXT], "topic")

# %% stage: model | Ekstraksi
# note: TF-IDF per dokumen dan isu korpus.
# uses: modeling/keyword_extract
kw = tfidf_keywords(docs, K, ngram=(1, 2))
print("isu korpus:", ctfidf_terms(docs, np.zeros(len(docs), int), 20)[0])
for t, k in list(zip(df[TEXT], kw))[:5]:
    print(k, "|", str(t)[:70])

# %% stage: out | Output
# note: Satu baris per dokumen.
# uses: 
df["keywords"] = ["; ".join(x) for x in kw]
df.to_csv("submission.csv", index=False)
