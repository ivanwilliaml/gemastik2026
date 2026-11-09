v, m = fit_hash_sgd(texts, y, bits=16, batch=100); assert m.predict_proba(v.transform(texts_test)).shape == (len(texts_test), 3)
