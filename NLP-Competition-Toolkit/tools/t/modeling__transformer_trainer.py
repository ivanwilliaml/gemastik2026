if MODELS.get("indobertweet"):
    m, t = train_transformer(MODELS["indobertweet"], texts[:48], y[:48], texts[48:64], y[48:64], batch=4, eval_every=6, out="r1", max_len=32)
    p = predict_transformer(m, t, texts_test[:8], max_len=32); assert p.shape == (8, 3) and abs(p.sum(1) - 1).max() < 1e-3
    del m; torch.cuda.empty_cache()
