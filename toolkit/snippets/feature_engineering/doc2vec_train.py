# title: Doc2Vec
# when: Eksperimen bila ada waktu.
# avoid: Pilihan pertama: hasil tidak stabil dan infer_vector lambat.
# tags: doc2vec dokumen
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class D2VVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim=200, epochs=10, min_count=2, workers=4, seed=42):
        self.dim, self.epochs, self.min_count, self.workers, self.seed = dim, epochs, min_count, workers, seed
    def _t(self, t):
        return re.findall(r"\w+", str(t).lower())
    def fit(self, texts, y=None):
        docs = [TaggedDocument(self._t(t), [i]) for i, t in enumerate(texts)]
        self.m_ = Doc2Vec(docs, vector_size=self.dim, epochs=self.epochs, min_count=self.min_count, dm=0, workers=self.workers, seed=self.seed)
        return self
    def transform(self, texts):
        return np.array([self.m_.infer_vector(self._t(t)) for t in texts], dtype="float32")
