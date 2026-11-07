E_ = np.random.rand(len(texts), 8); v = PrecomputedVec(list(texts), E_); assert v.fit(texts).transform(texts[:2]).shape == (2, 8)
assert make_head().fit(E_, y).predict_proba(E_).shape[1] == 3
if MODELS.get("sentiment"):
    p, n = pretrained_scores(MODELS["sentiment"], ["makanannya enak sekali"], batch=2); assert p.shape[1] == len(n) == 3
