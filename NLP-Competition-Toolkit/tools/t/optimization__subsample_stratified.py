i = subsample_idx(y, 90); assert len(i) == 90 and abs(np.mean(y[i] == y[0]) - np.mean(y == y[0])) < .1
