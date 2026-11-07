from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(texts); y2 = y.copy(); y2[200:] = -1; assert make_self_training(.6).fit(X, y2).predict(X).shape == y.shape
