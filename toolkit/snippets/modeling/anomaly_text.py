# title: Anomali: IForest, kNN, LOF, rekonstruksi, injeksi
# when: Tanpa label. Validasi dengan injeksi, seed tuning dan evaluasi dipisah. Gabungkan dengan rata-rata peringkat.
# avoid: Skor yang hanya mengikuti panjang teks: cek korelasi.
# tags: anomali isolation_forest knn lof rekonstruksi injeksi
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

def iforest_score(Z, seed=42):
    return -IsolationForest(n_estimators=200, random_state=seed, n_jobs=4).fit(Z).score_samples(Z)

def knn_score(Z, k=10):
    d, _ = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z).kneighbors(Z)
    return d[:, 1:].mean(1)

def lof_score(Z, k=20):
    m = LocalOutlierFactor(n_neighbors=k, metric="cosine").fit(Z)
    return -m.negative_outlier_factor_

def recon_score(X, k=100, seed=42):
    """Sisa rekonstruksi: |x|^2 - |proyeksi|^2. Aman untuk sparse besar, tanpa membuat matriks padat."""
    Zk = TruncatedSVD(k, random_state=seed).fit_transform(X)
    sq = np.asarray(X.multiply(X).sum(1)).ravel() if hasattr(X, "multiply") else (np.asarray(X) ** 2).sum(1)
    return np.maximum(sq - (Zk ** 2).sum(1), 0)

def rank_avg(*scores):
    return np.mean([rankdata(s) / len(s) for s in scores], 0)

def inject_eval(texts, make_scores, pool, share=.02, seed=42):
    """make_scores(list teks) -> skor. pool: teks asing yang disisipkan. Kembalikan ROC-AUC dan recall@share."""
    rng = np.random.default_rng(seed)
    n = max(1, int(len(texts) * share))
    ins = list(rng.choice(pool, n))
    s = np.asarray(make_scores(list(texts) + ins))
    lab = np.r_[np.zeros(len(texts)), np.ones(n)]
    top = np.argsort(-s)[:n]
    return {"auc": round(roc_auc_score(lab, s), 4), "recall@share": round(lab[top].mean(), 4)}
