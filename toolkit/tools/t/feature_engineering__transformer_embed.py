if MODELS.get("indobertweet"):
    assert mean_pool_embed(MODELS["indobertweet"], texts[:6], batch_size=2).shape[0] == 6
