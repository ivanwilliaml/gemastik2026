# title: Grup duplikat: hash, MinHash, komponen graf
# when: Sebelum fold. Copypasta sedikit berubah. Ukuran grup juga bisa menjadi fitur.
# avoid: Threshold rendah pada MinHash: kelompok terlalu besar.
# tags: minhash duplikat group graf copypasta
import re
import numpy as np
import pandas as pd

def norm_key(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()

def exact_groups(texts):
    return pd.factorize(pd.Series(list(texts)).map(norm_key))[0]

def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]; x = p[x]
    return x

def components(n, pairs):
    """Union-find. pairs: iterable (i, j). Kembalikan id grup per baris."""
    p = list(range(n))
    for a, b in pairs:
        p[_find(p, a)] = _find(p, b)
    return pd.factorize(np.array([_find(p, i) for i in range(n)]))[0]

def minhash_groups(texts, threshold=.8, num_perm=64, shingle=5):
    from datasketch import MinHash, MinHashLSH
    lsh, pairs = MinHashLSH(threshold=threshold, num_perm=num_perm), []
    for i, t in enumerate(map(norm_key, texts)):
        m = MinHash(num_perm=num_perm)
        for j in range(max(len(t) - shingle + 1, 1)):
            m.update(t[j:j + shingle].encode())
        pairs += [(i, int(k)) for k in lsh.query(m)]
        lsh.insert(str(i), m)
    return components(len(texts), pairs)

def pair_groups(a, b):
    """Grup untuk task pasangan: dua teks yang berbagi kalimat masuk satu komponen."""
    keys = pd.factorize(pd.Series([norm_key(t) for t in a] + [norm_key(t) for t in b]))[0]
    n = len(a)
    p = components(keys.max() + 1, zip(keys[:n], keys[n:]))
    return p[keys[:n]]

def group_size(groups):
    return pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
