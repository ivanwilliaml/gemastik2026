g = np.arange(len(y)) % 9; f = folds_group(g, 3); assert all(not set(g[a]) & set(g[b]) for a, b in f)
assert len(folds_leave_source(np.arange(12) % 3)) == 3
