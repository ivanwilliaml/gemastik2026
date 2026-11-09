# title: Metrik klaster dan akurasi tercocokkan
# when: Klaster. matched_accuracy bila ada ground truth untuk audit.
# avoid: Silhouette tanpa sample_size. Memilih k dari ground truth bila tidak diizinkan.
# tags: silhouette dbi ch ari nmi hungarian
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

def cluster_report(X, labels, truth=None, sample=10000, seed=42):
    mask = np.asarray(labels) >= 0
    X, labels = np.asarray(X)[mask], np.asarray(labels)[mask]
    r = dict(n_klaster=len(set(labels)), silhouette=silhouette_score(X, labels, sample_size=min(sample, len(labels)), random_state=seed), dbi=davies_bouldin_score(X, labels), ch=calinski_harabasz_score(X, labels))
    if truth is not None:
        t = np.asarray(truth)[mask]
        r.update(ari=adjusted_rand_score(t, labels), nmi=normalized_mutual_info_score(t, labels), matched_acc=matched_accuracy(t, labels))
    return {k: round(float(v), 4) for k, v in r.items()}

def matched_accuracy(truth, labels):
    """Akurasi setelah memasangkan klaster ke kelas (algoritma Hungaria)."""
    a, b = np.unique(truth, return_inverse=True)[1], np.unique(labels, return_inverse=True)[1]
    c = np.zeros((a.max() + 1, b.max() + 1), int)
    np.add.at(c, (a, b), 1)
    r, s = linear_sum_assignment(-c)
    return c[r, s].sum() / len(a)
