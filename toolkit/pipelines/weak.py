# title: Weak supervision
# for: weak
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
GOLD = "gold"                                    # kolom label manual kecil, -1 bila kosong
LEXICON = {"negatif": ["basi", "buruk", "kecewa", "jelek"], "positif": ["enak", "bagus", "puas", "suka"]}
VEC, MODEL = "tfidf_word", "lr"

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)

# %% stage: prep | Preprocessing
# note: Ganti preset bila perlu.
# uses: preprocessing/preprocess_all
texts = preprocess_corpus(df[TEXT], "tfidf")

# %% stage: model | Aturan menjadi label
# note: Audit tiap aturan.
# uses: semi_supervised/weak_supervision
H = keyword_lf(texts, LEXICON)
samples = audit_lf(H, texts, 30)
weak = vote_lf(H)
print("cakupan", round(float((weak != -1).mean()), 3))

# %% stage: bert | Latih classifier
# note: Generalisasi di luar kata kunci.
# uses: feature_engineering/make_vectorizer, modeling/make_model
m_ = weak != -1
vec = make_vectorizer(VEC)
clf = make_model(MODEL).fit(vec.fit_transform([t for t, k in zip(texts, m_) if k]), weak[m_])
pred = clf.predict(vec.transform(texts))

# %% stage: eval | Nilai pada gold
# note: Gold tidak pernah dipakai menulis aturan.
# uses: evaluation/metrics_classification
if GOLD in df:
    g = df[GOLD].to_numpy()
    ok = g != -1
    print("aturan", lf_accuracy(weak[ok], g[ok]), "\nclassifier", cls_report(g[ok], pred=pred[ok]))

# %% stage: out | Output
# note: Label per baris.
# uses: 
df["pred"] = np.array(list(LEXICON))[pred]
df.to_csv("submission.csv", index=False)
