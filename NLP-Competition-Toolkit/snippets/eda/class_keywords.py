# title: Kata khas per kelas dan uji asal label
# when: Mendeteksi artefak dan label buatan mesin. Skor hampir sempurna dari kata saja berarti label dibuat aturan.
# tags: keyword kelas artefak asal_label
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

def class_keywords(texts, y, k=15, min_df=3):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    X = v.fit_transform(texts)
    m = LogisticRegression(C=4.0, max_iter=2000).fit(X, y)
    n = v.get_feature_names_out()
    coef = m.coef_ if m.coef_.shape[0] > 1 else np.vstack([-m.coef_[0], m.coef_[0]])
    return {c: [n[i] for i in coef[j].argsort()[::-1][:k]] for j, c in enumerate(m.classes_)}

def label_origin_test(texts, y, max_features=300, cv=3):
    """Latih dari sedikit kata. Macro-F1 > 0.95 mencurigakan: label kemungkinan dari aturan."""
    X = TfidfVectorizer(max_features=max_features, binary=True).fit_transform(texts)
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv)
    return round(f1_score(y, p, average="macro"), 4)
