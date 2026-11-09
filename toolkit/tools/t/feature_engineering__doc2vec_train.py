v = D2VVectorizer(dim=10, epochs=1); assert v.fit(texts[:80]).transform(texts_test[:3]).shape == (3, 10)
