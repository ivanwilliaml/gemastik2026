start_clock(); assert clock("x") < 1
start_clock("13:00"); start_clock()
e = [("tfidf_union", "lr"), ("tfidf_union", "svc"), ("tfidf_word", "cnb")]
assert scale_plan(5000, e, 5) == (e, 5)
m, f = scale_plan(150000, e, 5); assert f == 3 and all(v == "tfidf_word" for v, _ in m)
b, f = scale_plan(500000, e, 5); assert b[0] == ("hashing", "sgd") and f == 3
assert scale_plan(500000, e, 5, force=True) == (e, 5)
assert len(sample_idx(10, 100)) == 10 and len(sample_idx(1000, 100)) == 100
