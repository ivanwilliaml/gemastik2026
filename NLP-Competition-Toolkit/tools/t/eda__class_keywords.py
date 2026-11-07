k = class_keywords(texts, y, 5); assert set(k) == set(np.unique(y)) and len(k[0]) == 5
assert 0 < label_origin_test(texts, y) <= 1
