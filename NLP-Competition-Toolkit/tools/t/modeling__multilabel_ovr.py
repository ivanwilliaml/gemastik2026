from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(ml_texts)
assert make_ovr(n_jobs=1).fit(X, Y).predict_proba(X).shape == Y.shape and make_chain().fit(X, Y).predict_proba(X).shape == Y.shape
