# title: Label per token: NER, POS, term aspek, keyphrase
# for: ner pos ate keyphrase
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

TRAIN, VALID, TEST = "train.txt", "valid.txt", "test.txt"
SEP = "\t"
USE_BERT = False
BERT_MODEL, MAX_LEN, BATCH = MODELS / "indobertweet", 128, 16

# %% stage: load | Muat data
# note: Token + tag per baris, kalimat dipisah baris kosong.
# uses: data_loading/load_conll
ts, tt = read_conll(TRAIN, sep=SEP)
vs, vt = read_conll(VALID, sep=SEP)
te, _ = read_conll(TEST, sep=SEP)
print(len(ts), len(vs), len(te))

# %% stage: eda | EDA
# note: Sensus tag dan panjang.
# uses: 
from collections import Counter
print(Counter(x for t in tt for x in t).most_common(12))
print("panjang kalimat p50/p95", np.quantile([len(s) for s in ts], [.5, .95]))

# %% stage: model | Baseline & CRF
# note: Tag terbanyak, lalu CRF fitur bentuk kata.
# uses: modeling/ner_baseline, modeling/ner_transformer, evaluation/metrics_ner
base = most_frequent_tag(ts, tt)
print("tag terbanyak", span_f1(vt, [repair_bio(x) for x in base(vs)])["micro"])
crf = fit_crf(ts, tt, iters=50)
vp = [repair_bio(x) for x in crf(vs)]
print("CRF", span_f1(vt, vp)["micro"])
predict = crf

# %% stage: bert | Transformer
# note: Lewati bila USE_BERT=False.
# uses: 
if USE_BERT:
    model, tok = train_token_model(BERT_MODEL, ts, tt, batch=BATCH, max_len=MAX_LEN)
    print("transformer", span_f1(vt, [repair_bio(x) for x in predict_tokens(model, tok, vs, max_len=MAX_LEN)])["micro"])
    predict = lambda s: predict_tokens(model, tok, s, max_len=MAX_LEN)
else:
    print("USE_BERT=False")

# %% stage: eval | Error analysis
# note: Jenis kesalahan span dan skor per tipe.
# uses: 
pv = [repair_bio(x) for x in predict(vs)]
print(span_f1(vt, pv))
print(span_errors(vt, pv))
print("akurasi token", round(float(np.mean([a == b for g, p in zip(vt, pv) for a, b in zip(g, p)])), 4))

# %% stage: out | Output
# note: Format sama dengan input.
# uses: 
with open("submission.txt", "w", encoding="utf-8") as f:
    for s, t in zip(te, predict(te)):
        f.write("\n".join(f"{a}{SEP}{b}" for a, b in zip(s, repair_bio(t))) + "\n\n")
print("selesai")
