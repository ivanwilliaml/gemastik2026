if MODELS.get("e5"):
    m = fit_setfit(MODELS["e5"], texts[:12], (y[:12] > 0).astype(int) if (y[:12] > 0).sum() > 1 else np.arange(12) % 2, batch=4, num_iterations=1)
    assert predict_setfit(m, texts[:3]).shape == (3, 2); del m
