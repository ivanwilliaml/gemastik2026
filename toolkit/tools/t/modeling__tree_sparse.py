from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
X = TruncatedSVD(16, random_state=1).fit_transform(TfidfVectorizer().fit_transform(texts))
a, b = np.arange(len(y)) % 4 != 0, np.arange(len(y)) % 4 == 0
m = fit_lgbm_es(make_lgbm(50), X[a], y[a], X[b], y[b], 5); assert m.predict_proba(X[b]).shape[1] == 3
m = make_xgb(20, patience=3).fit(X[a], y[a], eval_set=[(X[b], y[b])], verbose=False); assert m.predict(X[b]).shape[0] == b.sum()
