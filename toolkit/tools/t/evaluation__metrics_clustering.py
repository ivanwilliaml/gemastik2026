X_ = np.r_[np.zeros((10, 2)), np.ones((10, 2))] + np.random.default_rng(1).normal(0, .01, (20, 2)); l_ = np.r_[np.ones(10), np.zeros(10)].astype(int)
r = cluster_report(X_, l_, truth=1 - l_); assert r["matched_acc"] == 1 and r["ari"] == 1 and r["silhouette"] > .9
