# title: Deteksi bahasa dan kata penanda Jawa/Sunda
# when: Multibahasa. Bahasa daerah lebih andal dengan kata penanda.
# avoid: Teks sangat pendek.
# tags: bahasa langdetect jawa sunda
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

JAWA = {"ora", "ndak", "ndk", "mboten", "wes", "sing", "aku", "kowe", "piye", "opo", "madak"}
SUNDA = {"teu", "henteu", "atuh", "euy", "abdi", "naon", "kumaha", "mah"}

def make_lang_clf(max_features=200_000):
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=max_features, sublinear_tf=True), LogisticRegression(C=10, max_iter=2000))

def marker_lang(text):
    w = set(re.findall(r"\w+", str(text).lower()))
    j, s = len(w & JAWA), len(w & SUNDA)
    return "jv" if j > s and j >= 1 else "su" if s > j and s >= 1 else "lain"

def fast_detect(texts):
    """Pustaka fast-langdetect; butuh file model bawaan paket."""
    import fast_langdetect
    return [fast_langdetect.detect(str(t).replace("\n", " ")) for t in texts]
