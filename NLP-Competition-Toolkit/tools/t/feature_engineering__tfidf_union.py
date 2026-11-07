v = make_tfidf_union(); X = v.fit_transform(texts); assert X.shape[1] > 100 and v.transform(texts_test).shape[1] == X.shape[1]
