assert make_tfidf_char().fit_transform(texts).shape[0] == len(texts)
