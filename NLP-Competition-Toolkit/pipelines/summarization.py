# title: Ringkasan
# for: summarization
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
DOC, REF, N_SENT = "document", "summary", 2
USE_T5, T5_MODEL = False, MODELS / "t5-small-sum"   # profil all

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: eda | EDA
# note: Rasio panjang dan tanda baca.
# uses: 
print("rasio", round(float((train[REF].str.len() / train[DOC].str.len()).median()), 3), "| kalimat per dokumen", train[DOC].str.count(r"[.!?]").median())

# %% stage: model | Ekstraktif
# note: Lead-N sering sulit dikalahkan pada berita.
# uses: modeling/summarize_extractive, evaluation/metrics_rouge
docs, refs = train[DOC].fillna("").tolist(), train[REF].fillna("").tolist()
print("lead", rouge_report([lead_n(d, N_SENT) for d in docs], refs), "\ntextrank", rouge_report([textrank(d, N_SENT) for d in docs], refs))
pred = [lead_n(d, N_SENT) for d in test[DOC].fillna("")]

# %% stage: bert | T5
# note: Lewati bila USE_T5=False. Konfirmasi aturan seq2seq.
# uses: modeling/seq2seq
if USE_T5:
    model, tok = load_seq2seq(T5_MODEL)
    print("t5", rouge_report(generate_texts(model, tok, docs[:200]), refs[:200]))
    pred = generate_texts(model, tok, test[DOC].fillna("").tolist())
else:
    print("USE_T5=False")

# %% stage: out | Submission
# note: Kolom terakhir contoh submission.
# uses: postprocessing/label_mapping
sub = write_submission(SAMPLE, pred)
