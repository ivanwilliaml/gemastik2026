assert make_rnn("lstm", False, epochs=1, device="cpu", max_len=32).fit(texts, y).predict(texts_test).shape[0] == len(texts_test)
assert make_rnn(epochs=1, device="cpu", max_len=32).fit(texts, y).predict(texts_test).shape[0] == len(texts_test)
