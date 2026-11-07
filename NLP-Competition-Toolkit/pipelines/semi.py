# title: Semi-supervised
# for: pseudo selftrain confidence iterative
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
TEXT, TARGET = "text", "label"                   # baris tanpa label: NaN
VEC, MODEL, THRESHOLD, ROUNDS = "tfidf_word", "lr", .9, 2

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: prep | Preprocessing
# note: Ganti preset bila perlu.
# uses: preprocessing/preprocess_all
lab = train[train[TARGET].notna()]
texts = preprocess_corpus(lab[TEXT], "tfidf")
unl = preprocess_corpus(train[train[TARGET].isna()][TEXT], "tfidf")
test_texts = preprocess_corpus(test[TEXT], "tfidf")
print("berlabel", len(lab), "| tanpa label", len(unl))

# %% stage: split | Validasi pada gold
# note: Gold tidak pernah dilatih. Skor hanya dari label asli.
# uses: postprocessing/label_mapping
y, classes = encode_labels(lab[TARGET])
ix = np.random.default_rng(SEED).permutation(len(y))
g, tr = ix[:len(y) // 5], ix[len(y) // 5:]

# %% stage: model | Pseudo-label berulang
# note: Berhenti otomatis bila tak melebihi min_gain.
# uses: feature_engineering/make_vectorizer, modeling/make_model, semi_supervised/iterative_pseudo, semi_supervised/confidence_filter
vec, model, hist = iterative_pseudo(make_vectorizer(VEC), make_model(MODEL), [texts[i] for i in tr], y[tr], unl, [texts[i] for i in g], y[g], rounds=ROUNDS, threshold=THRESHOLD)
print("macro-F1 per putaran", hist)
p = model.predict_proba(vec.transform(test_texts))
print("baris sangat yakin", len(select_confident(p, THRESHOLD)[0]), "dari", len(p))

# %% stage: out | Submission
# note: Validasi format.
# uses: 
sub = write_submission(SAMPLE, classes[p.argmax(1)], allowed=classes)
