# title: preprocess(): satu fungsi lengkap dengan preset
# when: Satu fungsi untuk semua: unicode, HTML, URL, email, telepon, mention, hashtag, emoji, angka, huruf kecil, huruf berulang, tanda baca, slang, negasi, stopword, stem, spasi.
# avoid: Preset 'topic' sebelum classifier: uji lokal menunjukkan buang stopword merugikan.
# tags: preprocess cleaning unifikasi email telepon url html emoji slang stopword stem preset lowercase unicode
import html
import re
import unicodedata

SLANG = {"yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "tdk": "tidak", "bgt": "banget", "tp": "tapi", "klo": "kalau", "kalo": "kalau",
         "udh": "sudah", "dah": "sudah", "dpt": "dapat", "jg": "juga", "jd": "jadi", "bnyk": "banyak", "karna": "karena", "krn": "karena", "utk": "untuk", "jgn": "jangan",
         "org": "orang", "dr": "dari", "lgi": "lagi", "lg": "lagi", "bnr": "benar", "sdh": "sudah", "dgn": "dengan", "aja": "saja"}
STOPWORDS = set("""ada adalah agar akan aku anda antara apa apakah atau bagi bahwa banyak beberapa begitu bila bisa dalam dan dari dengan di dia dua hal hanya harus ia ini itu
jadi jika juga kami kamu karena ke kepada ketika kita lagi lain lebih maka masih mereka mungkin nya oleh pada para saat saja sama sampai sangat satu saya sebagai sebuah sedang
sehingga sekali selalu semua seperti setelah sudah supaya tapi telah tentang tersebut tetapi untuk yaitu yang""".split())   # sengaja tanpa tidak, bukan, jangan, belum, kurang
NEGATION = "tidak|tak|bukan|jangan|belum|kurang|tanpa"
_MOJIBAKE = re.compile("[\u00c3\u00c2\u00e2\u00f0]")

PRESETS = {
    # keep = biarkan, token = ganti penanda, remove = hapus
    "none":        dict(unicode=False, html=False, url="keep", email="keep", phone="keep", mention="keep", hashtag="keep", emoji="keep", number="keep", lower=False, repeat=False, punct="keep", slang=False, negation=False, stopword=False),
    "tfidf":       dict(unicode=True, html=True, url="token", email="token", phone="token", mention="token", hashtag="keep", emoji="keep", number="keep", lower=True, repeat=True, punct="keep", slang=False, negation=False, stopword=False),
    "social":      dict(unicode=True, html=True, url="token", email="token", phone="token", mention="token", hashtag="word", emoji="text", number="keep", lower=True, repeat=True, punct="keep", slang=True, negation=False, stopword=False),
    "topic":       dict(unicode=True, html=True, url="remove", email="remove", phone="remove", mention="remove", hashtag="word", emoji="remove", number="remove", lower=True, repeat=True, punct="remove", slang=True, negation=False, stopword=True),
    "transformer": dict(unicode=True, html=True, url="keep", email="keep", phone="keep", mention="keep", hashtag="keep", emoji="keep", number="keep", lower=False, repeat=False, punct="keep", slang=False, negation=False, stopword=False),
    "indobertweet": dict(unicode=True, html=True, url="HTTPURL", email="keep", phone="keep", mention="@USER", hashtag="keep", emoji="text", number="keep", lower=True, repeat=False, punct="keep", slang=False, negation=False, stopword=False),
}

def _sub(text, pattern, how, token):
    return text if how == "keep" else re.sub(pattern, " " if how == "remove" else f" {token if how == 'token' else how} ", text)

def preprocess(text, preset="tfidf", slang_dict=SLANG, stopwords=STOPWORDS, **override):
    """preprocess('Halo http://a.id', 'social'). Ubah satu langkah: preprocess(t, 'tfidf', emoji='remove')."""
    c = {**PRESETS[preset], **override}
    t = "" if text is None or text != text else str(text)
    if c["unicode"]:
        if _MOJIBAKE.search(t):                         # teks rusak encoding
            import ftfy
            t = ftfy.fix_text(t)
        t = unicodedata.normalize("NFKC", t).replace("​", "")
    if c["html"]:
        t = re.sub(r"<[^>]+>", " ", html.unescape(t))
    t = _sub(t, r"[\w.+-]+@[\w-]+\.[\w.-]+", c["email"], "EMAIL")          # email sebelum mention
    t = _sub(t, r"https?://\S+|www\.\S+", c["url"], "URL")
    t = _sub(t, r"(?<!\d)(?:\+?62|0)8\d{7,11}(?!\d)", c["phone"], "TELEPON")
    t = _sub(t, r"@\w+", c["mention"], "USER")
    if c["hashtag"] == "word":
        t = re.sub(r"#(\w+)", r" \1 ", t)
    elif c["hashtag"] != "keep":
        t = _sub(t, r"#\w+", c["hashtag"], "HASHTAG")
    if c["emoji"] != "keep":
        import emoji
        if c["emoji"] == "id":                          # emoji jadi kata Indonesia
            from indoNLP.preprocessing import emoji_to_words
            t = emoji_to_words(t, delimiter=(" ", " "))
        else:
            t = emoji.demojize(t, delimiters=(" ", " ")) if c["emoji"] == "text" else emoji.replace_emoji(t, " " if c["emoji"] == "remove" else " EMOJI ")
    t = _sub(t, r"\d+(?:[.,]\d+)*", c["number"], "ANGKA")
    if c["lower"]:
        keep = {m: m for m in ("HTTPURL", "@USER", "URL", "EMAIL", "TELEPON", "USER", "ANGKA", "EMOJI", "HASHTAG") if m in t}
        t = t.lower()
        for m in keep:
            t = re.sub(r"(?<!\w)" + re.escape(m.lower()) + r"(?!\w)", m, t)
    if c["repeat"]:
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    if c["punct"] == "remove":
        t = re.sub(r"[^\w\s]", " ", t)
    if c["slang"]:
        t = re.sub(r"\b\w+\b", lambda m: slang_dict.get(m.group(0).lower(), m.group(0)), t)
    if c["negation"]:
        t = re.sub(rf"\b({NEGATION})\s+(\w+)", r"\1_\2", t, flags=re.I)
    if c["stopword"]:
        t = " ".join(w for w in t.split() if w.lower() not in stopwords)
    return re.sub(r"\s+", " ", t).strip()

def preprocess_corpus(texts, preset="tfidf", stem=False, slang_csv=None, **override):
    """Versi cepat: teks identik hanya diproses sekali, stem per kata unik. Kembalikan list."""
    slang = dict(SLANG)
    if slang_csv:
        import pandas as pd
        lex = pd.read_csv(slang_csv)
        slang.update(dict(zip(lex["slang"].astype(str), lex["formal"].astype(str))))
    texts = list(texts)
    cache = {t: preprocess(t, preset, slang_dict=slang, **override) for t in {x if x == x else None for x in texts}}
    out = [cache[t if t == t else None] for t in texts]
    if stem:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        st = StemmerFactory().create_stemmer()
        m = {w: st.stem(w) for w in {w for d in out for w in d.split()}}
        out = [" ".join(m[w] for w in d.split()) for d in out]
    return out

def text_versions(df, text="text", slang_csv=None):
    """Tiga versi agar efek normalisasi bisa diuji: text_raw, text_clean (makna utuh), text_norm (slang dinormalkan)."""
    d = df.copy()
    d["text_raw"] = d[text]
    d["text_clean"] = preprocess_corpus(d[text], "tfidf")
    d["text_norm"] = preprocess_corpus(d[text], "social", slang_csv=slang_csv)
    return d
