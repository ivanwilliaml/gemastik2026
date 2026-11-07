v = FastTextVectorizer(dim=20, epochs=1); assert v.fit(texts).transform(["katabaruzzz"]).any()
open("v.vec", "w", encoding="utf-8").write("2 2\nmakan 1 0\nnasi 0 1\n"); assert load_vec("v.vec")["nasi"][1] == 1
