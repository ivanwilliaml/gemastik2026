if MODELS.get("t5"):
    m, t = load_seq2seq(MODELS["t5"]); assert len(generate_texts(m, t, ["ringkas kalimat ini."], max_new=8, beams=1)) == 1
assert callable(train_seq2seq)
