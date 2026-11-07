a, w1, w2, p_ = adversarial_auc(texts, texts_test); assert 0 <= a <= 1 and len(w1) == 20 and len(p_) == len(texts) + len(texts_test)
a2, *_ = adversarial_auc(["pesta bola seru"] * 30, ["nasi goreng enak"] * 30); assert a2 == 1
