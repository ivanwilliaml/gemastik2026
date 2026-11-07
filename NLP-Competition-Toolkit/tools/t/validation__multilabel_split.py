f = folds_multilabel(Y, 3); assert sum(len(b) for _, b in f) == len(Y)
assert all(abs(Y[b].mean(0) - Y.mean(0)).max() < .12 for _, b in f)
