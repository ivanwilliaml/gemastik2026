if MODELS.get("indobertweet"):
    m, t = train_multilabel(MODELS["indobertweet"], ml_texts[:24], Y[:24], batch=4, out="r2", max_len=32)
    assert predict_multilabel(m, t, ml_texts[:4], max_len=32).shape == (4, Y.shape[1]); del m; torch.cuda.empty_cache()
