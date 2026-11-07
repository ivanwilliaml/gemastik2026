from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(texts)
for m in (make_lr(), make_svc(), make_svc(calibrated=True), make_sgd(), make_ridge_clf(), make_dummy()):
    assert len(m.fit(X, y).predict(X)) == len(y)
assert make_lr().fit(X, y).score(X, y) > .7
