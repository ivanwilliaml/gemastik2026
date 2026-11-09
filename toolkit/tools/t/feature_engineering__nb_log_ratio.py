from sklearn.feature_extraction.text import CountVectorizer
X = CountVectorizer(binary=True).fit_transform(texts); assert NBRatio().fit(X, (y == 2).astype(int)).transform(X).shape == X.shape
