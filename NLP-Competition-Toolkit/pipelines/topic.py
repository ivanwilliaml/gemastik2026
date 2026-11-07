# title: Topic modeling
# for: topic
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
KS = (5, 10, 20)                                 # kandidat jumlah topik
METHOD = "nmf"                                   # nmf, lda
USE_BERTOPIC, E5 = False, MODELS / "e5-small"

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)

# %% stage: eda | EDA
# note: Termasuk kata teratas.
# uses: eda/eda_report, eda/vocab_stats
report = eda_report(df, None, TEXT)

# %% stage: clean | Cleaning data
# note: Copypasta tidak boleh menjadi topik.
# uses: noisy_data/clean_all
d = clean_dataset(df, TEXT, near_dup=False)

# %% stage: prep | Preprocessing
# note: Preset topic membuang stopword. Hanya di sini stopword boleh dibuang.
# uses: preprocessing/preprocess_all
docs = preprocess_corpus(d[TEXT], "topic")
print(top_ngrams(docs, 20))

# %% stage: model | Model & pilih k
# note: Coherence, diversity, dan baca isinya.
# uses: modeling/topic_nmf_lda, evaluation/metrics_topic, modeling/topic_embedding
fit = fit_nmf if METHOD == "nmf" else fit_lda
rows = []
for k in KS:
    _, w, _ = fit(docs, k=k, min_df=2)
    rows.append((k, npmi_coherence(w, docs)[0], topic_diversity(w)))
print(pd.DataFrame(rows, columns=["k", "npmi", "diversity"]).to_string())
best_k = max(rows, key=lambda r: r[1] + r[2])[0]
lab, words, W = fit(docs, k=best_k, min_df=2)
if USE_BERTOPIC:
    E = encode_st(E5, docs, cache="emb_topic.npy", prefix="passage: ")
    lab, topic_model = bertopic_sample(docs, E)
    words = [ctfidf_terms(docs, lab, 10)[t] for t in sorted(set(lab[lab >= 0]))]

# %% stage: eval | Baca topik
# note: Selalu baca: angka saja tidak cukup.
# uses: 
for i, w in enumerate(words):
    ex = [docs[j][:70] for j in np.where(lab == i)[0][:2]]
    print(i, int((lab == i).sum()), w[:8], ex)

# %% stage: out | Output
# note: Label per baris unik dan tabel topik.
# uses: 
d["topik"] = lab
d.to_csv("submission.csv", index=False)
pd.DataFrame({"topik": range(len(words)), "kata": [", ".join(w) for w in words]}).to_csv("topik.csv", index=False)
