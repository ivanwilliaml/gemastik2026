if MODELS.get("e5"):
    s = biencoder_cos(MODELS["e5"], ["makan nasi"], ["makan nasi"], batch_size=2); assert s[0] > .99
