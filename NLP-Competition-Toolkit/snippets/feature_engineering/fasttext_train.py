# title: FastText sendiri dan vektor .vec
# when: Pretrained dilarang dengan banyak slang, atau muat cc.id.300.vec dengan batas kata.
# avoid: File .bin: 7,2 GB.
# tags: fasttext oov embedding
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText, KeyedVectors

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim=200, window=5, min_count=2, epochs=5, workers=4, seed=42):
        self.dim, self.window, self.min_count, self.epochs, self.workers, self.seed = dim, window, min_count, epochs, workers, seed
    def _t(self, texts):
        return [re.findall(r"\w+", str(t).lower()) for t in texts]
    def fit(self, texts, y=None):
        self.ft_ = FastText(self._t(texts), vector_size=self.dim, window=self.window, min_count=self.min_count,
                            min_n=3, max_n=5, epochs=self.epochs, workers=self.workers, seed=self.seed).wv
        return self
    def transform(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, ws in enumerate(self._t(texts)):
            if ws:
                out[i] = self.ft_[ws].mean(0)
        return out

def load_vec(path, limit=200_000):
    """KeyedVectors dari cc.id.300.vec(.gz). `limit` membatasi kata agar hemat RAM."""
    return KeyedVectors.load_word2vec_format(path, binary=False, limit=limit)
