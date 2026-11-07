# title: Fitur pasangan teks
# when: NLI, parafrase, duplikat, konsistensi judul-isi. Wajib: bag of words gabungan sering tak berguna.
# avoid: Lupa arah (a,b) dan (b,a).
# tags: pair jaccard overlap kosinus fuzzy angka negasi entitas konsistensi
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

NEG = {"tidak", "tak", "bukan", "belum", "jangan", "tanpa", "kurang", "gak", "ga"}

def _w(t):
    return re.findall(r"\w+", str(t).lower())

def pair_features(a, b, vectorizer=None, window=30):
    """Kembalikan DataFrame fitur dan vectorizer (fit di data latih, pakai ulang di test)."""
    v = vectorizer or TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(list(a) + list(b))
    A, B = v.transform(a), v.transform(b)
    rows = []
    for x, y_ in zip(a, b):
        p, q = _w(x), _w(y_)
        P, Q = set(p), set(q)
        inter = len(P & Q)
        nx, ny = {w for w in p if w.isdigit()}, {w for w in q if w.isdigit()}
        ex, ey = set(re.findall(r"\b[A-Z][a-z]+\b", str(x))), set(re.findall(r"\b[A-Z][a-z]+\b", str(y_)))
        best = max([len(P & set(q[i:i + window])) for i in range(0, max(len(q) - window, 0) + 1, max(window // 2, 1))] or [0])
        rows.append(dict(
            len_a=len(p), len_b=len(q), len_diff=abs(len(p) - len(q)), jaccard=inter / max(len(P | Q), 1),
            cover_a=inter / max(len(P), 1), cover_b=inter / max(len(Q), 1), best_window=best / max(len(P), 1),
            fuzz_set=fuzz.token_set_ratio(x, y_) / 100, fuzz_part=fuzz.partial_ratio(x, y_) / 100,
            num_miss=len(nx - ny), num_conflict=float(bool(nx) and bool(ny) and not (nx & ny)),
            ent_miss=len(ex - ey), neg_diff=abs(len(NEG & P) - len(NEG & Q))))
    d = pd.DataFrame(rows).astype("float32")
    d["cos_tfidf"] = np.asarray(A.multiply(B).sum(1)).ravel()
    return d, v

def mutate_negatives(texts, seed=42, pool=None):
    """Bikin negatif sintetis untuk konsistensi: ganti angka atau entitas. Kembalikan teks termutasi."""
    rng = np.random.default_rng(seed)
    pool = pool or ["Jakarta", "Bandung", "Surabaya", "Medan", "Garut"]
    out = []
    for t in texts:
        if re.search(r"\d", t):
            out.append(re.sub(r"\d+", lambda m: str(int(m.group()) + int(rng.integers(1, 9))), t, count=1))
        elif re.search(r"\b[A-Z][a-z]+\b", t):
            out.append(re.sub(r"\b[A-Z][a-z]+\b", str(rng.choice(pool)), t, count=1))
        else:
            w = t.split()
            rng.shuffle(w)
            out.append(" ".join(w))
    return out
