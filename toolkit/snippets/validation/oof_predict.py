# title: oof_predict: inti mix and match
# when: Satukan vectorizer apa pun, model apa pun, dan fold apa pun. Vectorizer di-fit di dalam fold.
# tags: oof cv mix pipeline
import numpy as np
from sklearn.base import clone

def _proba(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    if hasattr(m, "decision_function"):
        raise ValueError('Margin bukan probabilitas. Gunakan predict untuk F1 atau CalibratedClassifierCV untuk probability ensemble.')
    return np.asarray(m.predict(X)).reshape(-1, 1)

def oof_predict(make_vec, make_model, texts, y, folds, test_texts=None, sample_weight=None, fit_kw=None):
    """make_vec dan make_model: objek atau fungsi tanpa argumen. folds: list (idx_train, idx_valid).
    Kembalikan (oof, test). Klasifikasi: probabilitas. Regresi: kolom tunggal. test = rata-rata fold."""
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y)
    oof, test = None, None
    for tr, va in folds:
        vec = make_vec() if callable(make_vec) and not hasattr(make_vec, "fit") else clone(make_vec)
        model = make_model() if callable(make_model) and not hasattr(make_model, "fit") else clone(make_model)
        Xtr, Xva = vec.fit_transform(list(texts[tr]), y[tr]), vec.transform(list(texts[va]))
        kw = dict(fit_kw or {})
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight)[tr]
        model.fit(Xtr, y[tr], **kw)
        p = _proba(model, Xva)
        classes = np.unique(y) if y.ndim == 1 and hasattr(model, 'classes_') else None
        if classes is not None:
            aligned = np.zeros((len(va),len(classes)))
            aligned[:,np.searchsorted(classes,model.classes_)] = p
            p = aligned
        if oof is None:
            oof = np.full((len(y), p.shape[1]), np.nan, dtype="float32")
        oof[va] = p
        if test_texts is not None:
            t = _proba(model, vec.transform(list(test_texts)))
            if classes is not None:
                aligned = np.zeros((len(t),len(classes)))
                aligned[:,np.searchsorted(classes,model.classes_)] = t
                t = aligned
            test = t / len(folds) if test is None else test + t / len(folds)
    return oof, test
