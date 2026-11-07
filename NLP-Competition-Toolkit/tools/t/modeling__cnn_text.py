assert make_textcnn(epochs=1, device="cpu").fit(texts, y).predict(texts_test).shape[0] == len(texts_test)
