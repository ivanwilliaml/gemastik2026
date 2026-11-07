# title: HDBSCAN dan penetapan noise
# when: k otomatis, ada noise. Setelah reduksi dan pada sampel.
# avoid: Dimensi tinggi langsung, atau semua baris.
# tags: hdbscan noise klaster gabung
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def fit_hdbscan(Z, min_size=50):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=min_size, prediction_data=False).fit_predict(np.asarray(Z, dtype="float64"))

def assign_noise(Z, labels, k=5):
    """Masukkan noise (-1) dan baris tanpa label ke klaster tetangga."""
    labels = np.asarray(labels).copy()
    ok = labels >= 0
    if ok.sum() and (~ok).sum():
        labels[~ok] = KNeighborsClassifier(n_neighbors=min(k, int(ok.sum()))).fit(Z[ok], labels[ok]).predict(Z[~ok])
    return labels

def merge_small(labels, min_size=20, to=-1):
    s = np.asarray(labels).copy()
    v, c = np.unique(s, return_counts=True)
    s[np.isin(s, v[c < min_size])] = to
    return s
