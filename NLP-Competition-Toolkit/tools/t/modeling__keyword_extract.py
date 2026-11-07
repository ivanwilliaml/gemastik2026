assert len(tfidf_keywords(texts[:50], 2, min_df=1)) == 50 and len(ctfidf_terms(texts, y, 3)) == 3
assert rake_keywords("anak makan nasi di sekolah yang baru", {"di", "yang"}, 2)[0] == "anak makan nasi"
V = np.eye(3); assert mmr_select(np.array([1., 0, 0]), V, ["a", "b", "c"], 2)[0] == "a"
