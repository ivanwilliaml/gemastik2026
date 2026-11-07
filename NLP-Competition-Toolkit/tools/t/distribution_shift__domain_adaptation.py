from sklearn.feature_extraction.text import TfidfVectorizer
a, b = fit_on_union(TfidfVectorizer(), texts, texts_test); assert a.shape[1] == b.shape[1]
rng_ = np.random.default_rng(1); assert coral(rng_.random((50, 4)), rng_.random((50, 4)) * 3).shape == (50, 4)
