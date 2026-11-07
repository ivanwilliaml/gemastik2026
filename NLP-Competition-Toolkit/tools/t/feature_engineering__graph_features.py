d = graph_features(["a", "b"], ["b", "c"]); assert d.common[0] == 0 and d.deg_a[1] == 2 and d.common.tolist() == [0.0, 0.0]
d = graph_features(["a"], ["c"], ["a", "c"], ["b", "b"]); assert d.common[0] == 1
