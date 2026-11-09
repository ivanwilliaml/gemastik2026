# title: Prediksi kata berikutnya
# for: nextword
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
docs = df[TEXT].fillna("").astype(str).tolist()
print(len(docs))

# %% stage: split | Validasi per dokumen
# note: Bukan per kata: mencegah bocor.
# uses: validation/split_holdout
a, b = holdout(docs, stratify=False)

# %% stage: model | N-gram
# note: Backoff dari konteks terpanjang.
# uses: modeling/ngram_next_word
lm = NGramLM(3).fit([docs[i] for i in a])

# %% stage: eval | Top-k accuracy
# note: Berapa kali kata benar ada di K tebakan.
# uses: 
hit = n = 0
for i in b:
    w = re.findall(r"\w+", docs[i].lower())
    for j in range(2, len(w)):
        n += 1
        hit += w[j] in lm.topk(" ".join(w[:j]), K)
print("top-k accuracy", round(hit / max(n, 1), 4), "dari", n)
