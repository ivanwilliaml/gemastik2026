# title: Augmentasi teks ringan
# when: Label sedikit. Hanya di fold latih dan ukur gain berpasangan.
# avoid: Mengubah makna: negasi dan angka. Augmentasi di validasi.
# tags: augmentasi dropout swap informal
import numpy as np

KEEP = {"tidak", "tak", "bukan", "jangan", "belum", "kurang"}
INFORMAL = {"yang": "yg", "tidak": "gak", "banget": "bgt", "tapi": "tp", "kalau": "klo", "sudah": "udh", "dengan": "dgn", "untuk": "utk", "juga": "jg"}

def word_dropout(text, p=.1, rng=None):
    rng = rng or np.random.default_rng()
    w = str(text).split()
    out = [x for x in w if x.lower() in KEEP or x.isdigit() or rng.random() > p]
    return " ".join(out or w)

def word_swap(text, n=1, rng=None):
    rng = rng or np.random.default_rng()
    w = str(text).split()
    for _ in range(n if len(w) > 1 else 0):
        i = rng.integers(0, len(w) - 1)
        w[i], w[i + 1] = w[i + 1], w[i]
    return " ".join(w)

def to_informal(text, p=.5, rng=None):
    rng = rng or np.random.default_rng()
    return " ".join(INFORMAL[x.lower()] if x.lower() in INFORMAL and rng.random() < p else x for x in str(text).split())

def augment_train(texts, y, fn=word_dropout, copies=1, seed=42):
    """Kembalikan (teks+augmentasi, y+salinan). Panggil hanya pada indeks latih tiap fold."""
    rng = np.random.default_rng(seed)
    t, yy = list(texts), list(y)
    for _ in range(copies):
        t += [fn(x, rng=rng) for x in texts]; yy += list(y)
    return t, np.array(yy)
