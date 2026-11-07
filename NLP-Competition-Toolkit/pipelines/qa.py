# title: Question answering
# for: qa
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
Q, C, A = "question", "context", "answer"
USE_BERT = False
BERT_MODEL, MAX_LEN, STRIDE = MODELS / "indobert-qa", 384, 128

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())

# %% stage: eda | EDA
# note: Panjang dan tipe pertanyaan.
# uses: 
print(train[C].str.split().str.len().quantile([.5, .95]).to_dict(), train[A].astype(str).str.split().str.len().quantile([.5, .95]).to_dict())
print(train[Q].str.split().str[0].str.lower().value_counts().head(8).to_dict())
print("jawaban ada di konteks", round(float(np.mean([str(a).lower() in str(c).lower() for a, c in zip(train[A], train[C])])), 3))

# %% stage: model | Baseline ekstraktif
# note: Kalimat terbaik dengan TF-IDF.
# uses: modeling/qa_extractive, evaluation/metrics_qa
base = [answer_tail(q, best_sentence(q, c)) for q, c in zip(train[Q], train[C])]
print("baseline", qa_report(base, train[A].tolist()))
pred = [answer_tail(q, best_sentence(q, c)) for q, c in zip(test[Q], test[C])]

# %% stage: bert | Model span
# note: Lewati bila USE_BERT=False. Decode kombinasi start-end dengan stride.
# uses: modeling/qa_transformer
if USE_BERT:
    tok = AutoTokenizer.from_pretrained(str(BERT_MODEL), local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(str(BERT_MODEL), local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")
    tr, score = qa_predict(model, tok, train[Q].tolist(), train[C].tolist(), max_len=MAX_LEN, stride=STRIDE)
    print("model span", qa_report(tr, train[A].tolist()))
    pred, score = qa_predict(model, tok, test[Q].tolist(), test[C].tolist(), max_len=MAX_LEN, stride=STRIDE)
else:
    print("USE_BERT=False")

# %% stage: out | Submission
# note: Kolom terakhir contoh submission.
# uses: postprocessing/label_mapping
sub = write_submission(SAMPLE, pred)
