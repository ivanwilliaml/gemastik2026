# title: clean_dataset(): satu fungsi cleaning data latih
# when: Sebelum split: kosong, pseudo-null, duplikat, near-duplicate, label bertentangan, baris tak bermakna, dan grup untuk fold.
# avoid: Membuang baris test. Mengubah label di fold validasi.
# tags: cleaning dedup duplikat konflik group near_duplicate minhash pseudo_null
import re
import numpy as np
import pandas as pd

_PSEUDO = {"", "nan", "none", "null", "-", "--", "n/a", "na", "[sticker]"}

def _groups_minhash(keys, threshold=.8, num_perm=64, shingle=5):
    from datasketch import MinHash, MinHashLSH
    lsh, par = MinHashLSH(threshold=threshold, num_perm=num_perm), list(range(len(keys)))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, t in enumerate(keys):
        m = MinHash(num_perm=num_perm)
        for j in range(max(len(t) - shingle + 1, 1)):
            m.update(t[j:j + shingle].encode())
        for k in lsh.query(m):
            par[find(i)] = find(int(k))
        lsh.insert(str(i), m)
    return pd.factorize(np.array([find(i) for i in range(len(keys))]))[0]

def clean_dataset(df, text="text", target=None, dedup=True, near_dup=False, conflict="majority", drop_empty=True, near_threshold=.8):
    """Kembalikan df bersih dengan kolom _group untuk fold. conflict: 'majority', 'drop', atau None."""
    n0 = len(df)
    df = df.copy()
    df[text] = df[text].fillna("").astype(str).str.strip()
    if drop_empty:
        df = df[~df[text].str.lower().isin(_PSEUDO)]
    key = df[text].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df["_group"] = pd.factorize(key)[0]
    if near_dup:
        df["_group"] = _groups_minhash(key.tolist(), near_threshold)
    n_conf = 0
    if target is not None and conflict:
        g = df.groupby(key)[target]
        bad = g.transform("nunique") > 1
        n_conf = int(bad.sum())
        if conflict == "majority":
            df[target] = g.transform(lambda s: s.mode().iloc[0])
        elif conflict == "drop":
            df, key = df[~bad], key[~bad]
    if dedup:
        keep = ~key.duplicated()
        df = df[keep]
    print(f"baris {n0} -> {len(df)} | konflik label {n_conf} | grup {df._group.nunique()}")
    return df.reset_index(drop=True)

def clean_test(df, text="text"):
    """Test tidak dibuang: hanya isi kosong."""
    df = df.copy()
    df[text] = df[text].fillna("").astype(str)
    return df
