assert make_hashing(16).transform(texts).shape[1] == 2 ** 16
