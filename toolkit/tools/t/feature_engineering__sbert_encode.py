if MODELS.get("e5"):
    E = encode_st(MODELS["e5"], texts[:8], cache="e.npy", prefix="passage: ", batch_size=4); assert E.shape[0] == 8 and abs(np.linalg.norm(E[0]) - 1) < 1e-3
    assert encode_st("-", texts[:8], cache="e.npy").shape[0] == 8
