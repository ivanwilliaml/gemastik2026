# title: make_vectorizer(): satu pintu semua representasi
# when: Ganti nama, seluruh pipeline tetap jalan: tfidf_union, tfidf_word, tfidf_char, count, binary, hashing, lsa, meta, tfidf_meta, w2v, w2v_idf, fasttext, doc2vec, embed, raw.
# avoid: Fit di data test tanpa izin. Fungsi ini dipakai di dalam fold.
# tags: vectorizer tfidf word2vec fasttext doc2vec embedding meta hashing lsa bow
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, Normalizer

def _tokens(t):
    return re.findall(r"\w+", str(t).lower())

def _meta(texts):
    s = pd.Series(list(texts)).fillna("").astype(str)
    n = s.str.len().clip(lower=1)
    return sparse.csr_matrix(pd.DataFrame({"len": np.log1p(n), "words": np.log1p(s.str.split().str.len()), "upper": s.str.count(r"[A-Z]") / n, "digit": s.str.count(r"\d") / n,
                                           "punct": s.str.count(r"[^\w\s]") / n, "excl": s.str.count("!").clip(upper=5), "url": s.str.count(r"https?://|URL"), "mention": s.str.count(r"@\w+|USER")}).to_numpy("float32"))

class RawText(BaseEstimator, TransformerMixin):
    """Teks mentah untuk model neural."""
    def fit(self, X, y=None): return self
    def transform(self, X): return list(X)

class GensimVec(BaseEstimator, TransformerMixin):
    """kind: w2v, fasttext, doc2vec. idf=True: rata-rata berbobot."""
    def __init__(self, kind="w2v", dim=200, window=5, min_count=2, epochs=5, idf=False, workers=4, seed=42):
        self.kind, self.dim, self.window, self.min_count, self.epochs, self.idf, self.workers, self.seed = kind, dim, window, min_count, epochs, idf, workers, seed
    def fit(self, X, y=None):
        from gensim.models import FastText, Word2Vec
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        toks = [_tokens(t) for t in X]
        kw = dict(vector_size=self.dim, min_count=self.min_count, epochs=self.epochs, workers=self.workers, seed=self.seed)
        if self.kind == "doc2vec":
            self.m_ = Doc2Vec([TaggedDocument(t, [i]) for i, t in enumerate(toks)], dm=0, **kw)
        else:
            self.m_ = (FastText if self.kind == "fasttext" else Word2Vec)(toks, window=self.window, **kw).wv
        self.w_ = None
        if self.idf:
            t = TfidfVectorizer(token_pattern=r"\w+").fit(X)
            self.w_ = dict(zip(t.get_feature_names_out(), t.idf_))
        return self
    def transform(self, X):
        out = np.zeros((len(X), self.dim), dtype="float32")
        for i, t in enumerate(X):
            w = _tokens(t)
            if self.kind == "doc2vec":
                out[i] = self.m_.infer_vector(w)
                continue
            w = w if self.kind == "fasttext" else [x for x in w if x in self.m_]
            if w:
                a = np.array([self.w_.get(x, 1.0) for x in w])[:, None] if self.w_ else np.ones((len(w), 1))
                out[i] = (self.m_[w] * a).sum(0) / a.sum()
        return out

class EmbedVec(BaseEstimator, TransformerMixin):
    """Embedding SentenceTransformer dengan cache per teks agar tiap fold tidak encode ulang."""
    _cache = {}
    def __init__(self, model_path=None, prefix="", batch=128):
        self.model_path, self.prefix, self.batch = model_path, prefix, batch
    def fit(self, X, y=None): return self
    def transform(self, X):
        c = EmbedVec._cache.setdefault(str(self.model_path), {})
        miss = [t for t in dict.fromkeys(map(str, X)) if t not in c]
        if miss:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(str(self.model_path), local_files_only=True)
            for t, v in zip(miss, m.encode([self.prefix + t for t in miss], batch_size=self.batch, normalize_embeddings=True, show_progress_bar=False)):
                c[t] = v.astype("float32")
        return np.vstack([c[str(t)] for t in X])

def make_vectorizer(kind="tfidf_union", **kw):
    """Semua hasilnya punya fit_transform dan transform."""
    word = dict(ngram_range=(1, 2), min_df=2, max_df=.95, sublinear_tf=True, dtype="float32", token_pattern=r"(?u)\b\w+\b")
    char = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=500_000, sublinear_tf=True, dtype="float32")
    if kind == "tfidf_word": return TfidfVectorizer(**{**word, **kw})
    if kind == "tfidf_char": return TfidfVectorizer(**{**char, **kw})
    if kind == "tfidf_union": return make_union(TfidfVectorizer(**word), TfidfVectorizer(**char))
    if kind == "count": return CountVectorizer(**{**dict(ngram_range=(1, 1), min_df=2, max_df=.95), **kw})
    if kind == "binary": return CountVectorizer(**{**dict(ngram_range=(1, 2), min_df=2, binary=True), **kw})
    if kind == "hashing": return HashingVectorizer(**{**dict(n_features=2 ** 20, alternate_sign=False, ngram_range=(1, 2), dtype="float32"), **kw})
    if kind == "lsa": return make_pipeline(TfidfVectorizer(**word), TruncatedSVD(kw.pop("k", 256), random_state=42), Normalizer(copy=False))
    if kind == "meta": return FunctionTransformer(_meta, validate=False)
    if kind == "tfidf_meta": return make_union(TfidfVectorizer(**word), FunctionTransformer(_meta, validate=False))
    if kind in ("w2v", "fasttext", "doc2vec"): return GensimVec(kind, **kw)
    if kind == "w2v_idf": return GensimVec("w2v", idf=True, **kw)
    if kind == "embed": return EmbedVec(**kw)
    if kind == "raw": return RawText()
    raise ValueError(f"vectorizer tidak dikenal: {kind}")
