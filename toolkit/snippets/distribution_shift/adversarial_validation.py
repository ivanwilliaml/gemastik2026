# title: Adversarial validation
# when: Ada test. AUC di atas 0,7 layak diselidiki. Fitur pembeda menjelaskan penyebabnya.
# avoid: Data besar: selisih kecil tetap terdeteksi. Lihat efek ke skor.
# tags: adversarial shift auc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

def adversarial_auc(train, test, top=20, cv=3):
    """Kembalikan (AUC, kata pembeda test, kata pembeda train, p_test per baris train+test)."""
    t = list(train) + list(test)
    lab = np.r_[np.zeros(len(train)), np.ones(len(test))]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000)
    X = v.fit_transform(t)
    m = LogisticRegression(C=1.0, max_iter=1000)
    p = cross_val_predict(m, X, lab, cv=cv, method="predict_proba")[:, 1]
    co = m.fit(X, lab).coef_[0]
    n = v.get_feature_names_out()
    o = co.argsort()
    return round(roc_auc_score(lab, p), 4), [n[i] for i in o[::-1][:top]], [n[i] for i in o[:top]], p
