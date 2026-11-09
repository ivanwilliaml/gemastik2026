# title: NB-SVM log-count ratio
# when: Sentimen biner, fitur ringan. Fit di dalam fold.
# tags: nbsvm ratio sentimen
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NBRatio(BaseEstimator, TransformerMixin):
    """Pipeline: make_pipeline(CountVectorizer(binary=True), NBRatio(), LogisticRegression())"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        y = np.asarray(y)
        pos = np.asarray(X[y == 1].sum(0)).ravel() + self.alpha
        neg = np.asarray(X[y != 1].sum(0)).ravel() + self.alpha
        self.r_ = np.log((pos / pos.sum()) / (neg / neg.sum()))
        return self
    def transform(self, X):
        return X.multiply(self.r_).tocsr()
