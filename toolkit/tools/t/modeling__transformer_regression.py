if MODELS.get("indobertweet"):
    m, t, sc = train_regression(MODELS["indobertweet"], texts[:24], y[:24].astype(float), batch=4, out="r3", max_len=32)
    assert predict_regression(m, t, texts[:4], sc, max_len=32).shape == (4,); del m; torch.cuda.empty_cache()
