# title: Kata kunci: TF-IDF, c-TF-IDF, RAKE, MMR
# when: Kata kunci dokumen, nama klaster, dan isu utama.
# avoid: Tanpa stopword: kata umum mendominasi.
# tags: keyword cTFIDF rake mmr keybert
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def tfidf_keywords(docs, k=5, ngram=(1, 3), stop=None, min_df=2):
    v = TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_df=.7, stop_words=stop, sublinear_tf=True)
    X = v.fit_transform(docs).tocsr()
    n = v.get_feature_names_out()
    return [[n[j] for j in X[i].indices[X[i].data.argsort()[::-1][:k]]] for i in range(X.shape[0])]

def ctfidf_terms(docs, labels, top=10, stop=None, min_df=2):
    """Kata khas per kelompok: {label: [kata]}. Abaikan label -1."""
    labels = np.asarray(labels)
    ids = [l for l in np.unique(labels) if l != -1]
    joined = [" ".join(str(d) for d, a in zip(docs, labels) if a == l) for l in ids]
    v = CountVectorizer(stop_words=stop, min_df=1, token_pattern=r"(?u)\b[^\d\W]{2,}\b")
    X = TfidfTransformer(sublinear_tf=True).fit_transform(v.fit_transform(joined)).toarray()
    n = v.get_feature_names_out()
    return {l: [n[j] for j in X[i].argsort()[::-1][:top]] for i, l in enumerate(ids)}

def rake_keywords(doc, stop, k=5, max_len=3):
    w = re.findall(r"\w+", str(doc).lower())
    phrases, cur = [], []
    for x in w:
        if x in stop:
            phrases.append(cur); cur = []
        else:
            cur.append(x)
    phrases = [p for p in phrases + [cur] if 0 < len(p) <= max_len]
    freq, deg = Counter(), Counter()
    for p in phrases:
        for x in p:
            freq[x] += 1; deg[x] += len(p)
    sc = {" ".join(p): sum(deg[x] / freq[x] for x in p) for p in phrases}
    return sorted(sc, key=sc.get, reverse=True)[:k]

def mmr_select(doc_vec, cand_vecs, cands, k=5, diversity=.5):
    """KeyBERT-style. Semua vektor L2-normalized."""
    sim = cand_vecs @ doc_vec
    picked = [int(sim.argmax())]
    while len(picked) < min(k, len(cands)):
        red = (cand_vecs @ cand_vecs[picked].T).max(1)
        sc = (1 - diversity) * sim - diversity * red
        sc[picked] = -9
        picked.append(int(sc.argmax()))
    return [cands[i] for i in picked]
