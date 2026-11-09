g = np.arange(len(y)) % 50; f = folds_strat_group(y, g, 3); assert all(not set(g[a]) & set(g[b]) for a, b in f)
