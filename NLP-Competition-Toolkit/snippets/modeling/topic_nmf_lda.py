# title: NMF dan LDA
# when: NMF untuk baseline dan teks pendek. LDA untuk dokumen lebih panjang.
# avoid: LDA dengan input TF-IDF.
# tags: topik nmf lda
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def _top(comp, names, k):
    return [[names[i] for i in c.argsort()[::-1][:k]] for c in comp]

def fit_nmf(docs, k=20, top=10, min_df=5, max_df=.5, stop=None, seed=42):
    v = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop, sublinear_tf=True, token_pattern=r"(?u)\b[^\d\W]{2,}\b")
    X = v.fit_transform(docs)
    m = NMF(k, init="nndsvd", max_iter=400, random_state=seed)
    W = m.fit_transform(X)
    return W.argmax(1), _top(m.components_, v.get_feature_names_out(), top), W

def fit_lda(docs, k=20, top=10, min_df=5, max_df=.5, stop=None, seed=42):
    v = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop, token_pattern=r"(?u)\b[^\d\W]{2,}\b")
    X = v.fit_transform(docs)
    m = LatentDirichletAllocation(k, learning_method="online", batch_size=4096, random_state=seed)
    W = m.fit_transform(X)
    return W.argmax(1), _top(m.components_, v.get_feature_names_out(), top), W
