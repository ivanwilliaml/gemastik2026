# title: Under/oversampling di fold latih
# when: Data besar atau kelas sangat minor. Tanpa paket tambahan.
# avoid: Sebelum split. Probabilitas perlu dikoreksi.
# tags: undersampling oversampling
import numpy as np

def undersample(idx, y, ratio=3, seed=42):
    """Batasi tiap kelas maks ratio x kelas terkecil. Terapkan pada indeks latih saja."""
    idx, y, rng = np.asarray(idx), np.asarray(y), np.random.default_rng(seed)
    v, c = np.unique(y[idx], return_counts=True)
    cap = int(c.min() * ratio)
    return np.concatenate([rng.choice(idx[y[idx] == k], min(cap, n), replace=False) for k, n in zip(v, c)])

def oversample(idx, y, seed=42):
    idx, y, rng = np.asarray(idx), np.asarray(y), np.random.default_rng(seed)
    v, c = np.unique(y[idx], return_counts=True)
    return np.concatenate([rng.choice(idx[y[idx] == k], c.max(), replace=n < c.max()) for k, n in zip(v, c)])

def correct_prior(p, prior_train, prior_true):
    """Koreksi probabilitas setelah resampling (multikelas)."""
    q = np.asarray(p) * np.asarray(prior_true) / np.asarray(prior_train)
    return q / q.sum(1, keepdims=True)
