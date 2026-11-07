# title: Word2Vec sendiri dan rata-rata berbobot IDF
# when: Pretrained dilarang dan korpus cukup besar. Vectorizer ini kompatibel dengan oof_predict.
# avoid: Korpus kecil: lemah.
# tags: word2vec embedding idf
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def _toks(texts):
    return [re.findall(r"\w+", str(t).lower()) for t in texts]

class W2VVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim=200, window=5, min_count=2, epochs=5, idf=False, workers=4, seed=42):
        self.dim, self.window, self.min_count, self.epochs, self.idf, self.workers, self.seed = dim, window, min_count, epochs, idf, workers, seed
    def fit(self, texts, y=None):
        self.w2v_ = Word2Vec(_toks(texts), vector_size=self.dim, window=self.window, min_count=self.min_count,
                             epochs=self.epochs, workers=self.workers, seed=self.seed).wv
        self.idf_ = None
        if self.idf:
            t = TfidfVectorizer(token_pattern=r"\w+").fit(texts)
            self.idf_ = dict(zip(t.get_feature_names_out(), t.idf_))
        return self
    def transform(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, ws in enumerate(_toks(texts)):
            ws = [w for w in ws if w in self.w2v_]
            if ws:
                wt = np.array([self.idf_.get(w, 1.0) for w in ws])[:, None] if self.idf_ else np.ones((len(ws), 1))
                out[i] = (self.w2v_[ws] * wt).sum(0) / wt.sum()
        return out
