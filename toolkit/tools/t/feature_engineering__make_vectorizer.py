for k in ("tfidf_word", "tfidf_char", "tfidf_union", "count", "binary", "lsa", "meta", "tfidf_meta", "raw"):
    kw = dict(k=8) if k == "lsa" else {}
    v = make_vectorizer(k, **kw); a = v.fit_transform(texts); b = v.transform(texts_test); assert (len(a) if k == "raw" else a.shape[0]) == len(texts) and (len(b) if k == "raw" else b.shape[0]) == len(texts_test)
assert make_vectorizer("hashing", n_features=2 ** 12).transform(texts).shape[1] == 2 ** 12
for k in ("w2v", "w2v_idf", "fasttext", "doc2vec"):
    assert make_vectorizer(k, dim=8, epochs=1).fit(texts[:100]).transform(texts_test[:2]).shape == (2, 8)
if MODELS.get("e5"):
    v = make_vectorizer("embed", model_path=MODELS["e5"], prefix="passage: ", batch=4); assert v.fit_transform(texts[:4]).shape[0] == 4 and v.transform(texts[:2]).shape[0] == 2
