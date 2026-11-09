assert make_attn_gru(epochs=1, device="cpu", max_len=32).fit(texts, y).predict(texts_test).shape[0] == len(texts_test)
