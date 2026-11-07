from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(texts).astype("float64"); t = y.astype(float) * 3
assert make_ridge(log=True).fit(X, t).predict(X).min() > -1 and make_lgbm_reg("l1", 10).fit(X, t).predict(X).shape == t.shape and make_dummy_reg().fit(X, t).predict(X)[0] > 0
