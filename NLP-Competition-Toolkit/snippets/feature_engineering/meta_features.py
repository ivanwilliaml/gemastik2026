# title: Meta features teks dan waktu
# when: Spam, hoaks, buzzer, regresi engagement. Tree atau gabung ke sparse.
# avoid: Fitur yang bergeser di test: cek PSI dulu.
# tags: meta panjang kapital url emoji waktu burst inset leksikon
import re
import numpy as np
import pandas as pd

def meta_features(texts):
    s = pd.Series(list(texts)).fillna("").astype(str)
    n = s.str.len().clip(lower=1)
    return pd.DataFrame({
        "n_char": n, "n_word": s.str.split().str.len(),
        "upper_rt": s.str.count(r"[A-Z]") / n, "digit_rt": s.str.count(r"\d") / n,
        "punct_rt": s.str.count(r"[^\w\s]") / n, "excl": s.str.count("!"), "ques": s.str.count(r"\?"),
        "url": s.str.count(r"https?://|www\."), "mention": s.str.count(r"@\w+"), "hashtag": s.str.count(r"#\w+"),
        "uniq_rt": s.map(lambda t: len(set(t.split())) / max(len(t.split()), 1)),
    }).astype("float32")

def time_features(ts, events=None):
    """events: {'nama': 'YYYY-MM-DD'} menghasilkan jarak hari ke kejadian."""
    t = pd.to_datetime(pd.Series(list(ts)), errors="coerce")
    d = pd.DataFrame({"hour": t.dt.hour, "dow": t.dt.dayofweek, "month": t.dt.month})
    day = t.dt.floor("D")
    d["vol_hari"] = day.map(day.value_counts())
    for k, v in (events or {}).items():
        d["hari_ke_" + k] = (t - pd.Timestamp(v)).dt.days
    return d.astype("float32")

def lexicon_score(texts, pos, neg):
    """Skor leksikon sebagai fitur (InSet atau daftar sendiri), bukan pemberi label."""
    pos, neg = set(pos), set(neg)
    toks = [re.findall(r"\w+", str(t).lower()) for t in texts]
    return np.array([[sum(w in pos for w in x), sum(w in neg for w in x)] for x in toks], dtype="float32")
