from sklearn.feature_extraction.text import TfidfVectorizer as _T
from sklearn.decomposition import TruncatedSVD as _S
X = _T().fit_transform(texts); Z = _S(16, random_state=1).fit_transform(X)
for k in ("lr", "svc", "svc_cal", "sgd", "ridge_clf", "cnb", "mnb", "dummy"):
    assert len(make_model(k).fit(X, y).predict(X)) == len(y)
for k, kw in (("lgbm", dict(n_estimators=10)), ("xgb", dict(n_estimators=5)), ("rf", dict(n_estimators=5)), ("mlp", {}), ("smote_lr", {})):
    assert len(make_model(k, **kw).fit(Z, y).predict(Z)) == len(y)
for k in ("cnn", "gru", "bigru", "lstm", "bilstm", "attn"):
    assert make_model(k, epochs=1, device="cpu", max_len=24, batch=256).fit(texts, y).predict_proba(texts_test).shape == (len(texts_test), 3)
Xm = _T().fit_transform(ml_texts)
assert make_model("ovr").fit(Xm, Y).predict_proba(Xm).shape == Y.shape and make_model("chain").fit(Xm, Y).predict_proba(Xm).shape == Y.shape
t_ = y * 1.5 + 1
for k, kw in (("ridge", {}), ("ridge_log", {}), ("lgbm_reg", dict(n_estimators=10)), ("lgbm_l1", dict(n_estimators=10)), ("rf_reg", dict(n_estimators=5)), ("dummy_reg", {})):
    assert make_model(k, **kw).fit(X.astype("float64"), t_).predict(X.astype("float64")).shape == t_.shape
