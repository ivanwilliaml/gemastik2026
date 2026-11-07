# title: Persentil panjang dan max_length
# when: Selalu. max_len dari cakupan 95% pada 20.000 sampel, bukan 512.
# tags: panjang max_length token
import numpy as np
import pandas as pd

def length_table(texts, by=None):
    s = pd.Series(list(texts)).fillna("").astype(str)
    d = pd.DataFrame({"char": s.str.len(), "word": s.str.split().str.len()})
    if by is not None:
        return d.groupby(np.asarray(by)).quantile([.5, .9, .95]).unstack().round(0)
    return d.quantile([.01, .5, .9, .95, .99]).round(0)

def suggest_max_len(tokenizer, texts, n=20000, cover=.95, seed=42, upper=512):
    rng = np.random.default_rng(seed)
    texts = list(texts)
    idx = rng.choice(len(texts), min(n, len(texts)), replace=False)
    lens = [len(x) for x in tokenizer([texts[i] for i in idx], truncation=False)["input_ids"]]
    return int(min(np.quantile(lens, cover), upper))
