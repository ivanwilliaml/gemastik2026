# title: BM25 sparse
# when: Baseline retrieval tanpa paket tambahan.
# avoid: Query dan dokumen dinormalisasi berbeda.
# tags: bm25 retrieval
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

class BM25:
    def __init__(self, k1=1.5, b=.75, ngram=(1, 1)):
        self.k1, self.b, self.v = k1, b, CountVectorizer(ngram_range=ngram, token_pattern=r"(?u)\b\w+\b")
    def fit(self, docs):
        X = self.v.fit_transform(docs).astype("float32").tocsr()
        n, dl = X.shape[0], np.asarray(X.sum(1)).ravel()
        df_ = np.asarray((X > 0).sum(0)).ravel()
        self.idf_ = np.log(1 + (n - df_ + .5) / (df_ + .5))
        rows = np.repeat(np.arange(n), np.diff(X.indptr))
        d = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * dl[rows] / dl.mean()))
        self.M_ = sparse.csr_matrix((d * self.idf_[X.indices], X.indices, X.indptr), shape=X.shape)
        return self
    def scores(self, queries):
        return (self.v.transform(queries).astype("float32") > 0).astype("float32") @ self.M_.T
    def search(self, queries, k=10):
        """Kembalikan (idx, skor) top-k per query. Batch per 1.000 query pada korpus besar."""
        S = self.scores(queries).toarray()
        idx = np.argsort(-S, 1)[:, :k]
        return idx, np.take_along_axis(S, idx, 1)
