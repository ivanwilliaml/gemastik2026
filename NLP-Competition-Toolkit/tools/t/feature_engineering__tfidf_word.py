assert make_tfidf_word().fit_transform(texts).shape[0] == len(texts)
