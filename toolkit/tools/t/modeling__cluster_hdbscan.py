rng_ = np.random.default_rng(0); Z = np.r_[rng_.normal(0, .1, (50, 2)), rng_.normal(5, .1, (50, 2)), [[2.5, 2.5]]]
l = fit_hdbscan(Z, 10); assert len(set(l[l >= 0])) == 2
assert (assign_noise(Z, l) >= 0).all() and merge_small([0, 0, 1], 2).tolist() == [0, 0, -1]
