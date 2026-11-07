# title: TF-IDF karakter
# when: Slang, typo, OOV tinggi, multibahasa, teks pendek.
# avoid: Rentang lebar tanpa max_features: RAM.
# tags: tfidf char ngram oov
from sklearn.feature_extraction.text import TfidfVectorizer

def make_tfidf_char(ngram=(2, 5), min_df=3, max_features=500_000, analyzer="char_wb"):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_features=max_features,
                           sublinear_tf=True, dtype="float32")
