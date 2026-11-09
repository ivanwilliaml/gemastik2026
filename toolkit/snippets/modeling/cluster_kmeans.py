# title: MiniBatchKMeans, pencarian k, stabilitas, konsensus, Agglomerative
# when: Baseline klaster. L2 normalize agar setara kosinus.
# avoid: Silhouette penuh pada 500 ribu baris. Agglomerative tanpa sampel.
# tags: klaster kmeans stabilitas konsensus agglomerative spherical
import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

def kmeans_l2(X, k, seed=42):
    return MiniBatchKMeans(k, batch_size=4096, n_init=5, random_state=seed).fit_predict(normalize(X))

def search_k(X, ks=(5, 10, 20, 30, 50), sample=10000, seeds=(1, 2)):
    rows = []
    X = normalize(X)
    for k in ks:
        a, b = [kmeans_l2(X, k, s) for s in seeds]
        rows.append({"k": k, "silhouette": round(silhouette_score(X, a, sample_size=min(sample, X.shape[0]), random_state=1), 4), "stabilitas_ari": round(adjusted_rand_score(a, b), 4)})
    return rows

def consensus_kmeans(X, k, seeds=(1, 2, 3, 4, 5), sample=None):
    """Rata-rata ko-klaster antar seed, lalu Agglomerative. Untuk sampel kecil yang stabil."""
    L = np.array([kmeans_l2(X, k, s) for s in seeds])
    co = (L[:, :, None] == L[:, None, :]).mean(0)
    return AgglomerativeClustering(k, metric="precomputed", linkage="average").fit_predict(1 - co)

def agglo_sample(X, threshold=.5):
    return AgglomerativeClustering(None, distance_threshold=threshold, metric="cosine", linkage="average").fit_predict(np.asarray(X))
