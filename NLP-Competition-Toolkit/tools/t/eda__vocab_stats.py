o = vocab_stats(texts, texts_test); assert 0 <= o["oov_token"] <= 1 and o["vocab"] > 50
assert len(top_ngrams(texts, 5, (1, 2))) == 5
