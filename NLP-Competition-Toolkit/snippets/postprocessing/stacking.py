# title: Stacking dari OOF
# when: Hanya bila OOF rapi, terutama untuk log loss.
# avoid: Macro-F1 pada data kecil: bobot overfit.
# tags: stacking meta oof
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict

def stack_cls(oofs, y, tests, C=.1, cv=5):
    """oofs, tests: list array probabilitas. Kembalikan (oof stack jujur, prediksi test)."""
    X, T = np.hstack(oofs), np.hstack(tests)
    m = LogisticRegression(C=C, max_iter=1000)
    return cross_val_predict(m, X, y, cv=cv, method="predict_proba"), m.fit(X, y).predict_proba(T)

def stack_reg(oofs, y, tests, alpha=1.0, cv=5):
    X, T = np.column_stack(oofs), np.column_stack(tests)
    m = Ridge(alpha=alpha)
    return cross_val_predict(m, X, y, cv=cv), m.fit(X, y).predict(T)
