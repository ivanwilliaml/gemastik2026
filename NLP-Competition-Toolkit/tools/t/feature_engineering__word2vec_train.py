v = W2VVectorizer(dim=20, epochs=1, idf=True); assert v.fit(texts).transform(texts_test).shape == (len(texts_test), 20)
