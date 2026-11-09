# title: Aturan, lookup train, dan default baris kosong
# when: Pola pasti yang teruji di OOF, atau test berisi teks train dan aturan mengizinkan.
# avoid: Aturan dari sampel kecil. Menyetel dari test.
# tags: aturan lookup default kosong
import re
import numpy as np
import pandas as pd

def _k(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()

def lookup_train(train_texts, y, test_texts, pred, min_agree=1.0):
    """Salin label teks identik yang konsisten. Kembalikan (pred baru, jumlah diganti)."""
    d = pd.DataFrame({"k": [_k(t) for t in train_texts], "y": list(y)})
    g = d.groupby("k").y.agg(lambda s: s.mode().iloc[0] if s.value_counts(normalize=True).iloc[0] >= min_agree else None).dropna().to_dict()
    pred = np.array(pred, dtype=object)
    hit = np.array([_k(t) in g for t in test_texts])
    pred[hit] = [g[_k(t)] for t, h in zip(test_texts, hit) if h]
    return pred, int(hit.sum())

def default_empty(test_texts, pred, default):
    pred = np.array(pred, dtype=object)
    pred[pd.Series(list(test_texts)).fillna("").astype(str).str.strip().eq("").to_numpy()] = default
    return pred

def apply_rule(texts, pred, pattern, label):
    """Ganti prediksi bila regex cocok. Uji presisinya di OOF dulu."""
    pred = np.array(pred, dtype=object)
    pred[[bool(re.search(pattern, str(t), re.I)) for t in texts]] = label
    return pred
