assert label_table(y).n.sum() == len(y)
t, co = multilabel_table(Y); assert co.shape[0] == Y.shape[1]
assert target_table(np.arange(10.)).loc["max"] == 9
