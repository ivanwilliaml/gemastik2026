r = retrieval_report([[1, 2, 3]], [{2}], 2); assert r["p@2"] == .5 and r["mrr"] == .5 and r["map"] == .5 and abs(r["ndcg@2"] - 1 / np.log2(3)) < 1e-3
assert keyword_f1_at_k([["a", "b"]], [["A", "c"]]) == .5
