# title: Hashing + SGD partial_fit
# when: Jaring pengaman 500 ribu baris. Memori kecil, belajar per batch.
# avoid: Lupa classes pada partial_fit pertama.
# tags: skala hashing sgd partial_fit
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

def fit_hash_sgd(texts, y, bits=20, batch=20000, epochs=2, alpha=1e-6, seed=42):
    vec = HashingVectorizer(n_features=2 ** bits, alternate_sign=False, ngram_range=(1, 2), dtype="float32")
    m, y, classes, rng = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed), np.asarray(y), np.unique(y), np.random.default_rng(seed)
    texts = np.asarray(texts, dtype=object)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            idx = order[i:i + batch]
            m.partial_fit(vec.transform(texts[idx]), y[idx], classes=classes)
    return vec, m
