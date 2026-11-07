# title: Branch D: aman untuk transformer dan IndoBERTweet
# when: Semua fine-tuning dan embedding. IndoBERTweet dilatih dengan aturan ini.
# avoid: Stemming dan buang stopword.
# tags: indobertweet transformer sep
import re

def clean_transformer(text):
    """Pembersihan umum: hanya spasi dan karakter kontrol."""
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x08\x0b-\x1f]", " ", str(text))).strip()

def clean_indobertweet(text):
    import emoji
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", t)
    t = re.sub(r"@\w+", "@USER", t)
    t = emoji.demojize(t, delimiters=(" ", " "))
    return re.sub(r"\s+", " ", t).strip()

def join_columns(df, cols, sep=" [SEP] ", max_chars=None):
    """Gabung judul dan isi. max_chars membatasi tiap kolom agar satu kolom tidak menghabiskan max_len."""
    parts = [df[c].fillna("").astype(str) for c in cols]
    if max_chars:
        parts = [p.str[:max_chars] for p in parts]
    out = parts[0]
    for p in parts[1:]:
        out = out + sep + p
    return out.tolist()
