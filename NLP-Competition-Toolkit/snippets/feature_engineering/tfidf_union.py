# title: Gabungan TF-IDF kata dan karakter
# when: Baseline paling kuat dengan biaya rendah. Satu objek: fit di fold latih, transform di mana saja.
# avoid: .toarray(): tetap sparse.
# tags: tfidf union baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

def make_tfidf_union(word_max=300_000, char_max=500_000):
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.95, max_features=word_max, sublinear_tf=True, dtype="float32", token_pattern=r"(?u)\b\w+\b"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=char_max, sublinear_tf=True, dtype="float32"))
