# title: Clustering dokumen
# for: doc_cluster emb_cluster dimred repr
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
TRUTH = None                                     # kolom ground truth untuk audit, bila ada
KS = (3, 5, 10)
REPR = "lsa"                                     # lsa, w2v, fasttext, doc2vec, embed

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)

# %% stage: eda | EDA
# note: Profil dan kosakata.
# uses: eda/eda_report
report = eda_report(df, None, TEXT)

# %% stage: prep | Preprocessing
# note: Preset topic untuk tema. Transformer untuk embedding.
# uses: preprocessing/preprocess_all
docs = preprocess_corpus(df[TEXT], "topic")

# %% stage: feat | Representasi
# note: Satu nama di Setup.
# uses: feature_engineering/make_vectorizer
kw = dict(k=64) if REPR == "lsa" else dict(model_path=MODELS / "e5-small", prefix="passage: ") if REPR == "embed" else dict(dim=64)
Z = make_vectorizer(REPR, **kw).fit_transform(docs)
print(Z.shape)

# %% stage: model | Klaster
# note: Silhouette dan stabilitas antar seed.
# uses: modeling/cluster_kmeans
table = search_k(Z, KS)
print(pd.DataFrame(table).to_string())
k = max(table, key=lambda r: r["silhouette"] + r["stabilitas_ari"])["k"]
lab = kmeans_l2(Z, k)

# %% stage: eval | Evaluasi & baca isi
# note: Nama klaster dari kata khas.
# uses: evaluation/metrics_clustering, modeling/keyword_extract
print(cluster_report(Z, lab, None if TRUTH is None else df[TRUTH]))
for c, w in ctfidf_terms(docs, lab, 8).items():
    print(c, int((lab == c).sum()), w)

# %% stage: out | Output
# note: Label per baris.
# uses: 
df["cluster"] = lab
df.to_csv("submission.csv", index=False)
