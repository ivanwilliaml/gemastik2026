# title: Terjemahan dan normalisasi
# for: translation normalization
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
SRC, TGT = "source", "target"
USE_T5, T5_MODEL = False, MODELS / "t5-small-sum"

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: model | Baseline kamus
# note: Kamus berbatas kata, sering sulit dikalahkan.
# uses: preprocessing/preprocess_all, evaluation/metrics_seq2seq
src, tgt = train[SRC].fillna("").tolist(), train[TGT].fillna("").tolist()
base = preprocess_corpus(src, "none", slang=True)
print("salin apa adanya: CER", cer(src, tgt), "\nkamus: CER", cer(base, tgt), "WER", wer(base, tgt), "BLEU", bleu(base, tgt))
pred = preprocess_corpus(test[SRC], "none", slang=True)

# %% stage: bert | Fine-tune T5
# note: Lewati bila USE_T5=False.
# uses: modeling/seq2seq
if USE_T5:
    model, tok = train_seq2seq(T5_MODEL, src, tgt)
    pred = generate_texts(model, tok, test[SRC].fillna("").tolist())
else:
    print("USE_T5=False")

# %% stage: out | Submission
# note: Kolom terakhir contoh submission.
# uses: postprocessing/label_mapping
sub = write_submission(SAMPLE, pred)
