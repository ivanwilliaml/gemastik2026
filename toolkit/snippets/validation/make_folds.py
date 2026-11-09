# title: make_folds() dan oof_predict(): skema validasi dan inti mix and match
# when: kind: auto, stratified, kfold, regression, group, strat_group, multilabel, time, source. oof_predict menerima vectorizer dan model apa pun.
# avoid: Random fold bila ada duplikat lintas fold, atau waktu pada test.
# tags: fold validasi stratified group time multilabel oof overlap leakage
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

def _ml_folds(Y, n, seed):
    raw = np.asarray(Y, dtype=float)
    assert np.isin(raw[np.isfinite(raw)], [0, 1]).all(), "Target harus 0/1/NaN"
    Y, rng = np.hstack([raw == 1, np.isfinite(raw)]), np.random.default_rng(seed)
    left, fold = np.ones(len(Y), bool), -np.ones(len(Y), int)
    need, cap = np.tile(Y.sum(0) / n, (n, 1)).astype(float), np.full(n, len(Y) / n)
    while left.any():
        c = Y[left].sum(0)
        c[c == 0] = 10 ** 9
        lab = int(c.argmin())
        pool = np.where(left)[0] if c[lab] == 10 ** 9 else rng.permutation(np.where(left & Y[:, lab])[0])
        for i in pool:
            d = need[:, lab]
            cand = np.where(d == d.max())[0]
            f = int(cand[cap[cand].argmax()])
            fold[i], left[i] = f, False
            cap[f] -= 1
            need[f, Y[i]] -= 1
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n)]

def make_folds(y=None, groups=None, times=None, kind="auto", n=5, seed=42, valid_share=.2):
    """Kembalikan list (idx_train, idx_valid). time: satu holdout terbaru."""
    if kind == "auto":
        y_ = None if y is None else np.asarray(y)
        if times is not None: kind = "time"
        elif y_ is None: kind = "group" if groups is not None else "kfold"
        elif y_.ndim == 2: kind = "multilabel"
        elif np.issubdtype(y_.dtype, np.floating) and len(np.unique(y_)) > 20: kind = "regression" if groups is None else "group"
        else: kind = "strat_group" if groups is not None else "stratified"
        print("skema fold:", kind)
    if kind == "time":
        t = pd.to_datetime(pd.Series(list(times)), errors="coerce")
        cut = t.quantile(1 - valid_share)
        return [(np.where(t < cut)[0], np.where(t >= cut)[0])]
    if kind == "source": return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))
    N = len(y) if y is not None else len(groups)
    Z = np.zeros(N)
    if kind == "multilabel":
        if groups is None: return _ml_folds(y, n, seed)
        gid, inverse = np.unique(groups, return_inverse=True)
        raw = np.asarray(y, float)
        profile = np.array([np.r_[np.any(raw[inverse==i]==1,axis=0), np.any(np.isfinite(raw[inverse==i]),axis=0)] for i in range(len(gid))])
        return [(np.flatnonzero(np.isin(inverse,a)),np.flatnonzero(np.isin(inverse,b))) for a,b in _ml_folds(profile,n,seed)]
    if kind == "kfold": return list(KFold(n, shuffle=True, random_state=seed).split(Z))
    if kind == "group": return list(GroupKFold(n, shuffle=True, random_state=seed).split(Z, groups=groups))
    if kind == "strat_group": return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(Z, y, groups))
    if kind == "regression":
        v = np.asarray(y, float)
        return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(Z, np.digitize(v, np.unique(np.quantile(v, np.linspace(0, 1, 11)[1:-1])))))
    if kind == "stratified": return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(Z, y))
    raise ValueError(kind)

def check_folds(texts, folds):
    """Persen teks validasi yang juga ada di latih. Harus mendekati 0 kecuali test memang overlap."""
    k = np.array([re.sub(r"\W+", " ", str(t).lower()).strip() for t in texts], dtype=object)
    r = [round(float(np.isin(k[b], k[a]).mean()), 4) for a, b in folds]
    print("overlap per fold:", r)
    return r

def _proba(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    if hasattr(m, "decision_function"):
        raise ValueError('Margin bukan probabilitas. Gunakan predict untuk F1 atau classifier terkalibrasi untuk probability ensemble.')
    return np.asarray(m.predict(X)).reshape(-1, 1)

def oof_predict(make_vec, make_mdl, texts, y, folds, test_texts=None, sample_weight=None):
    """make_vec, make_mdl: fungsi tanpa argumen. Vectorizer di-fit di dalam fold. Kembalikan (oof, test rata-rata fold)."""
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)
    oof, test = None, None
    for tr, va in folds:
        vec, model = make_vec(), make_mdl()
        Xtr, Xva = vec.fit_transform(list(texts[tr]), y[tr]), vec.transform(list(texts[va]))
        kw = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)[tr]}
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
