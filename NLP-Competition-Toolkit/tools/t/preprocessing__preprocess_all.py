s = "<b>HALOOO</b> kirim ke budi@mail.id atau 081234567890 cek http://a.id @andi #MBG yg tidak enak 😀 harga 1.500"
o = preprocess(s, "social"); assert "EMAIL" in o and "TELEPON" in o and "URL" in o and "USER" in o and "haloo" in o and "yang" in o and "grinning" in o and "budi@mail" not in o and "mbg" in o and "<b>" not in o
o = preprocess(s, "topic"); assert "http" not in o and "tidak" in o and "yang" not in o and "1500" not in o and "@" not in o
o = preprocess(s, "indobertweet"); assert "HTTPURL" in o and "@USER" in o
assert preprocess(s, "none") == re.sub(r"\s+", " ", s).strip() and preprocess(None) == "" and preprocess(float("nan")) == ""
assert "tidak_enak" in preprocess(s, "tfidf", negation=True) and "harga ANGKA" in preprocess(s, "tfidf", number="token") and "EMOJI" in preprocess("a😀", "tfidf", emoji="token")
c = preprocess_corpus(["Memakan nasi", "Memakan nasi", None], "topic", stem=True); assert c[0] == c[1] == "makan nasi" and c[2] == ""
