# title: LGBMRanker
# when: Ada label relevansi dan fitur skor per pasangan query-dokumen.
# avoid: group tidak sesuai urutan baris. Fungsi ini mengurutkan otomatis.
# tags: ranker lambdarank lightgbm
import lightgbm as lgb
import numpy as np

def fit_ranker(X, rel, qid, n=300, lr=.05):
    qid = np.asarray(qid)
    order = np.argsort(qid, kind="stable")
    _, sizes = np.unique(qid[order], return_counts=True)
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=n, learning_rate=lr, verbose=-1, min_child_samples=1)
    m.fit(np.asarray(X)[order], np.asarray(rel)[order], group=sizes)
    return m
