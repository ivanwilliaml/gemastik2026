# title: ABSA: TF-IDF satu model per aspek
# when: Kolom per aspek, aspek sedikit, dan tanpa GPU.
# avoid: Aspek dengan sedikit data: gunakan prior.
# tags: absa aspek
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

def absa_cv(texts, A, test_texts=None, n=5, C=4.0, seed=42):
    """A: DataFrame satu kolom per aspek berisi kelas. Kembalikan (dict oof, dict test) label prediksi."""
    texts = np.asarray(texts, dtype=object)
    oof, test = {}, {}
    for col in A.columns:
        yc = A[col].astype(str).to_numpy()
        pred = np.empty(len(yc), dtype=object)
        k = min(n, int(np.unique(yc, return_counts=True)[1].min()))
        tp = []
        if k < 2:
            pred[:] = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(yc), 1)), yc).predict(np.zeros((len(yc), 1)))
            oof[col] = pred
            test[col] = None if test_texts is None else np.full(len(test_texts), pred[0], dtype=object)
            continue
        for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(texts, yc):
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            m = LogisticRegression(C=C, max_iter=2000).fit(v.fit_transform(texts[tr]), yc[tr])
            pred[va] = m.predict(v.transform(texts[va]))
            if test_texts is not None:
                tp.append(m.predict_proba(v.transform(test_texts)))
        oof[col] = pred
        if test_texts is not None:
            test[col] = m.classes_[np.mean(tp, 0).argmax(1)]
    return oof, test
