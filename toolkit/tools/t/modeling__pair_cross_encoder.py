if MODELS.get("indobertweet"):
    m, t = train_pair(MODELS["indobertweet"], pa[:24], pb[:24], py[:24], batch=4, out="r4", max_len=48)
    assert predict_pair(m, t, pa[:4], pb[:4], symmetric=True, max_len=48).shape == (4, 3); del m; torch.cuda.empty_cache()
