rng_ = np.random.default_rng(1); X_ = rng_.random((60, 3)); r_ = (X_[:, 0] > .5).astype(int); q_ = np.arange(60) % 6
assert fit_ranker(X_, r_, q_, 10).predict(X_).shape == (60,)
