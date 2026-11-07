rng_ = np.random.default_rng(1); assert "tidak" in word_dropout("makan tidak enak sekali nasi", .9, rng_)
assert sorted(word_swap("a b c", 1, rng_).split()) == ["a", "b", "c"] and to_informal("yang itu", 1, rng_) == "yg itu"
t_, y_ = augment_train(["a b", "c d"], [0, 1], copies=2); assert len(t_) == 6 and len(y_) == 6
