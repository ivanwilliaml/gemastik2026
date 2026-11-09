# title: ABSA: kolom per aspek
# for: absa
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
TEXT, ASPECTS = "text", ["a0", "a1", "a2"]
PREPROCESS = "tfidf"

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: eda | EDA
# note: Termasuk matriks aspek x sentimen untuk slide.
# uses: eda/eda_report
report = eda_report(train, test, TEXT)
print(train[ASPECTS].melt().groupby(["variable", "value"]).size().unstack(fill_value=0).to_string())

# %% stage: clean | Cleaning data
# note: Dedup.
# uses: noisy_data/clean_all
train, test = clean_dataset(train, TEXT, conflict=None), clean_test(test, TEXT)

# %% stage: prep | Preprocessing
# note: Ganti preset di Setup.
# uses: preprocessing/preprocess_all
texts, test_texts = preprocess_corpus(train[TEXT], PREPROCESS), preprocess_corpus(test[TEXT], PREPROCESS)

# %% stage: model | Model per aspek
# note: Validasi dan fitur sudah di dalam fungsi: satu TF-IDF + LR per aspek.
# uses: modeling/absa_per_aspect, evaluation/metrics_classification
oof, pred = absa_cv(texts, train[ASPECTS], test_texts, n=5)
scores = {a: round(f1_score(train[a].astype(str), oof[a], average="macro", zero_division=0), 4) for a in ASPECTS}
print(scores, "rata-rata", round(float(np.mean(list(scores.values()))), 4))

# %% stage: bert | Input aspek [SEP] teks
# note: Format panjang untuk fine-tune bila diperlukan.
# uses: modeling/absa_pair_input
long = absa_long(train.assign(**{TEXT: texts}), TEXT, ASPECTS)
print(long.head(3).to_string(), "\nlatih dengan texts=long.aspect, texts2=long.text")

# %% stage: out | Submission
# note: Isi tiap kolom aspek.
# uses: postprocessing/label_mapping
sub = pd.read_csv(SAMPLE)
for a in ASPECTS:
    sub[a] = pred[a]
sub.to_csv("submission.csv", index=False)
print(sub.head(3).to_string())
