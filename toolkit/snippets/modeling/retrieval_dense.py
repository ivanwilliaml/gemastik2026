# title: Retrieval dense, TF-IDF top-k, dan RRF hybrid
# when: Query berbeda kata dari dokumen. Hybrid saat dua sinyal sama kuat.
# avoid: Matriks n x n pada 500 ribu baris. Tanpa normalisasi L2.
# tags: dense retrieval topk rrf hybrid dokumen_serupa
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

def topk_dense(Q, D, k=10, batch=512):
    """Q dan D sudah L2-normalized. Kembalikan (idx, skor)."""
    D = np.asarray(D); idx, sc = [], []
    for i in range(0, len(Q), batch):
        S = np.asarray(Q[i:i + batch]) @ D.T
        p = np.argpartition(-S, min(k, D.shape[0] - 1), 1)[:, :k]
        s = np.take_along_axis(S, p, 1)
        o = np.argsort(-s, 1)
        idx.append(np.take_along_axis(p, o, 1)); sc.append(np.take_along_axis(s, o, 1))
    return np.vstack(idx), np.vstack(sc)

def topk_sparse(Q, D, k=10, batch=512):
    """TF-IDF kosinus top-k per batch."""
    idx, sc = [], []
    for i in range(0, Q.shape[0], batch):
        S = linear_kernel(Q[i:i + batch], D)
        p = np.argsort(-S, 1)[:, :k]
        idx.append(p); sc.append(np.take_along_axis(S, p, 1))
    return np.vstack(idx), np.vstack(sc)

def rrf_merge(rankings, k=60, top=10):
    """rankings: list matriks idx (n_query, n_kandidat). Reciprocal Rank Fusion."""
    out = []
    for q in range(rankings[0].shape[0]):
        s = {}
        for r in rankings:
            for pos, d in enumerate(r[q]):
                s[int(d)] = s.get(int(d), 0) + 1 / (k + pos + 1)
        out.append(sorted(s, key=s.get, reverse=True)[:top])
    return np.array(out)
