# title: Vocabulary gabungan dan CORAL
# when: Shift kuat dan aturan membolehkan memakai test tanpa label.
# avoid: Menyetel apa pun dari skor test.
# tags: domain coral vocabulary
import numpy as np
from scipy.linalg import inv, sqrtm

def fit_on_union(vectorizer, train, test):
    """Fit vocabulary di train+test bila diizinkan. Tidak memakai label."""
    vectorizer.fit(list(train) + list(test))
    return vectorizer.transform(train), vectorizer.transform(test)

def coral(Xs, Xt, reg=1.0):
    """Sejajarkan kovarians sumber ke target. Input padat, misal SVD atau embedding."""
    d = Xs.shape[1]
    Cs, Ct = np.cov(Xs, rowvar=False) + reg * np.eye(d), np.cov(Xt, rowvar=False) + reg * np.eye(d)
    return np.real(Xs @ inv(sqrtm(Cs)) @ sqrtm(Ct))
