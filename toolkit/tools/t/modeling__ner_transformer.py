assert repair_bio(["O", "I-A", "I-A", "I-B"]) == ["O", "B-A", "I-A", "B-B"]
assert vote_tags([[["O", "B-A"]], [["O", "I-A"]], [["O", "B-A"]]]) == [["O", "B-A"]]
if MODELS.get("indobertweet"):
    m, t = train_token_model(MODELS["indobertweet"], ner_s[:24], ner_t[:24], batch=4, epochs=1, max_len=48, out="r5")
    p = predict_tokens(m, t, ner_s[:3], max_len=48); assert [len(x) for x in p] == [len(x) for x in ner_s[:3]]; del m; torch.cuda.empty_cache()
