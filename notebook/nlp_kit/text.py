"""Pembersihan teks, pola yang diukur saat EDA, dan langkah preprocessing yang bisa dipilih."""
import html
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]{1,40}>")
BYTES_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
ANON_RE = re.compile(r"\b(USER|RT|URL)\b|\\n")
EMOJI = "\U0001F300-\U0001FAFF☀-➿"
EMOJI_RE = re.compile(f"[{EMOJI}]")
REPEAT_RE = re.compile(r"([a-z])\1{2,}")
LAUGH_RE = re.compile(r"\b(?:w*k+w+k*w*k*){2,}\b|\b(?:ha){2,}h?\b|\b(?:he){2,}h?\b|\b(?:hi){2,}h?\b", re.I)
HASHTAG_RE = re.compile(r"#(\w+)")
DIGIT_RE = re.compile(r"\d+")
SYMBOL_RE = re.compile(rf"[^\w\s{EMOJI}]")

PATTERNS = {
    "url": URL_RE,
    "mention": MENTION_RE,
    "hashtag": HASHTAG_RE,
    "emoji": EMOJI_RE,
    "emoji_byte": BYTES_RE,
    "html": TAG_RE,
    "token_anonim": ANON_RE,
    "angka": DIGIT_RE,
    "huruf_berulang": re.compile(r"([a-zA-Z])\1{2,}"),
    "tawa": LAUGH_RE,
    "tanda_seru_tanya": re.compile(r"[!?]{2,}"),
    "huruf_besar_semua": re.compile(r"\b[A-Z]{4,}\b"),
    "non_ascii": re.compile(r"[^\x00-\x7F]"),
}

# Pola yang kalau muncul membuat langkah preprocessing layak diuji.
STEP_TRIGGERS = {"karakter_ulang": "huruf_berulang", "tawa": "tawa", "hashtag": "hashtag", "angka": "angka"}

STOPWORDS = set(
    """yang dan di ke dari untuk pada dengan ini itu atau juga karena sudah akan ada saya kami kita mereka dia
    ia kamu anda oleh sebagai dalam para lalu agar bagi serta seperti saat kalau jika maka lebih sangat hanya
    bisa dapat harus telah masih namun tetapi tapi nya lah kah pun si sang dll yg dgn utk""".split()
)

FALLBACK_SLANG = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "tdk": "tidak", "gak": "tidak", "ga": "tidak", "nggak": "tidak",
    "ngga": "tidak", "gk": "tidak", "krn": "karena", "karna": "karena", "bgt": "banget", "aja": "saja", "udah": "sudah",
    "udh": "sudah", "sdh": "sudah", "blm": "belum", "kalo": "kalau", "klo": "kalau", "gmn": "bagaimana", "dr": "dari",
    "jg": "juga", "tp": "tapi", "sm": "sama", "org": "orang", "bs": "bisa", "dpt": "dapat", "pd": "pada", "lg": "lagi",
    "skrg": "sekarang", "emg": "memang", "gue": "saya", "gw": "saya", "lu": "kamu", "lo": "kamu",
}


@lru_cache(maxsize=4)
def load_slang(model_dir) -> dict[str, str]:
    """Kamus informal ke baku dari paket offline. Kalau tidak ada, pakai daftar kecil bawaan."""
    root = Path(model_dir).parent / "data"
    path = next(root.glob("**/colloquial-indonesian-lexicon.csv"), None) if root.exists() else None
    if path is None:
        print(f"kamus slang tidak ada di {root}, memakai daftar bawaan {len(FALLBACK_SLANG)} kata")
        return dict(FALLBACK_SLANG)
    lexicon = pd.read_csv(path)
    slang = dict(zip(lexicon["slang"].astype(str), lexicon["formal"].astype(str)))
    print(f"kamus slang: {len(slang)} kata")
    return slang


def _decode_bytes(match: re.Match) -> str:
    """Emoji yang tersimpan sebagai teks byte, misalnya \\xf0\\x9f\\x98\\xa0, dikembalikan."""
    raw = bytes.fromhex(match.group(0).replace("\\x", ""))
    return " " + raw.decode("utf-8", "ignore") + " "


def clean_text(text: str) -> str:
    """Langkah wajib: hanya membuang gangguan teknis. Emoji, tanda seru, dan slang tetap."""
    text = BYTES_RE.sub(_decode_bytes, str(text))
    text = TAG_RE.sub(" ", html.unescape(text))
    text = ANON_RE.sub(" ", URL_RE.sub(" ", MENTION_RE.sub(" ", text)))
    return SPACE_RE.sub(" ", text).strip().lower()


def make_steps(slang: dict[str, str]) -> dict:
    """Langkah opsional. Tambah langkah sendiri cukup dengan menambah isi kamus ini."""
    return {
        "karakter_ulang": lambda t: REPEAT_RE.sub(r"\1", t),
        "tawa": lambda t: LAUGH_RE.sub("wkwk", t),
        "hashtag": lambda t: HASHTAG_RE.sub(r"\1", t),
        "slang": lambda t: " ".join(slang.get(word, word) for word in t.split()),
        "angka": lambda t: DIGIT_RE.sub("0", t),
        "tanda_baca": lambda t: SPACE_RE.sub(" ", SYMBOL_RE.sub(" ", t)).strip(),
        "stopword": lambda t: " ".join(word for word in t.split() if word not in STOPWORDS),
    }


def apply_steps(texts, steps: dict, names) -> list[str]:
    """Jalankan beberapa langkah berurutan."""
    output = list(texts)
    for name in names:
        output = [steps[name](text) for text in output]
    return output


def pattern_rates(texts: pd.Series) -> pd.Series:
    """Persen baris yang mengandung tiap pola."""
    texts = texts.astype(str)
    return pd.Series({name: 100 * texts.str.contains(pattern).mean() for name, pattern in PATTERNS.items()})


def near_duplicate_groups(texts: list[str], threshold: float = 0.9) -> np.ndarray:
    """Teks yang kemiripan char TF-IDF-nya di atas threshold mendapat grup yang sama."""
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=min(6, len(texts)), metric="cosine").fit(vectors)
    distance, index = neighbors.kneighbors(vectors)
    parent = list(range(len(texts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (row_dist, row_idx) in enumerate(zip(distance, index)):
        for dist, j in zip(row_dist, row_idx):
            if dist <= 1 - threshold:
                parent[root(i)] = root(int(j))
    return np.array([root(i) for i in range(len(texts))])


def adversarial_scores(train_texts: list[str], test_texts: list[str]) -> tuple[float, np.ndarray]:
    """AUC yang membedakan dua kumpulan teks, serta skor kemiripan tiap baris dengan kumpulan kedua."""
    texts = list(train_texts) + list(test_texts)
    target = np.array([0] * len(train_texts) + [1] * len(test_texts))
    vectors = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores = cross_val_predict(model, vectors, target, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(target, scores)), scores


def adversarial_terms(train_texts: list[str], test_texts: list[str], top: int = 10) -> pd.DataFrame:
    """Kata yang paling membedakan dua kumpulan teks."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    vectors = vectorizer.fit_transform(list(train_texts) + list(test_texts))
    target = np.array([0] * len(train_texts) + [1] * len(test_texts))
    weights = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vectors, target).coef_[0]
    words = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(weights)
    return pd.DataFrame({"ciri_latih": words[order[:top]], "ciri_test": words[order[::-1][:top]]})
