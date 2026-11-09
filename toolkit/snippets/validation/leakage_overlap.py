# title: Cek overlap dan sampling test
# when: Setelah membuat fold dan saat EDA.
# tags: leakage overlap kebocoran test
import re
import numpy as np

def _key(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()

def overlap_rate(a, b):
    """Persen baris b yang teksnya ada di a."""
    A = {_key(t) for t in a}
    return round(float(np.mean([_key(t) in A for t in b])), 4)

def check_folds(texts, folds):
    texts = np.asarray(texts, dtype=object)
    return [overlap_rate(texts[a], texts[b]) for a, b in folds]
