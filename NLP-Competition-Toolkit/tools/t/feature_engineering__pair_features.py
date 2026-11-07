d, v = pair_features(["Budi makan 2 nasi", "kalimat sama"], ["Budi tidak makan 3 nasi", "kalimat sama"])
assert d.shape[0] == 2 and d.num_conflict[0] == 1 and d.neg_diff[0] == 1 and abs(d.cos_tfidf[1] - 1) < 1e-5
d2, _ = pair_features(["x"], ["y"], vectorizer=v); assert d2.shape[1] == d.shape[1]
assert mutate_negatives(["ada 5 orang"])[0] != "ada 5 orang"
