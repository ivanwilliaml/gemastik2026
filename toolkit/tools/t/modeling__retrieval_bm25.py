b_ = BM25().fit(["makan nasi goreng", "burung terbang", "nasi uduk enak"]); i, s = b_.search(["nasi goreng"], 2); assert i[0, 0] == 0 and s[0, 0] > s[0, 1] > 0
