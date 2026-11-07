from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(texts); assert make_nb().fit(X, y).score(X, y) > .5 and make_nb("m").fit(X, y)
