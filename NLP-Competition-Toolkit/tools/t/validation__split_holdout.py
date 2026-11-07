a, b = holdout(y); assert len(a) + len(b) == len(y) and not set(a) & set(b)
g = np.arange(len(y)) % 7; a, b = holdout(y, groups=g); assert not set(g[a]) & set(g[b])
