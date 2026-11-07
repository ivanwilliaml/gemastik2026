if MODELS.get("qa"):
    tok_ = AutoTokenizer.from_pretrained(MODELS["qa"], local_files_only=True); m_ = AutoModelForQuestionAnswering.from_pretrained(MODELS["qa"], local_files_only=True).to(DEV)
    a, sc = qa_predict(m_, tok_, ["Siapa presiden Indonesia pertama?"], ["Soekarno adalah presiden Indonesia pertama yang memimpin sejak 1945."], max_len=64, stride=16, batch=2); assert len(a) == 1 and isinstance(a[0], str); del m_
    m_, t_ = train_qa(MODELS["qa"], ["Siapa presiden?"] * 4, ["Soekarno adalah presiden."] * 4, ["Soekarno"] * 4, [0] * 4, max_len=32, stride=8, batch=2, out="r6"); del m_; torch.cuda.empty_cache()
