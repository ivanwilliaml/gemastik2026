# title: TF-IDF kata dan n-gram kata
# when: Langkah pertama hampir semua task dokumen.
# avoid: Stopword saat klasifikasi.
# tags: tfidf word ngram
from sklearn.feature_extraction.text import TfidfVectorizer

def make_tfidf_word(ngram=(1, 2), min_df=2, max_df=.95, max_features=None):
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df, max_features=max_features,
                           sublinear_tf=True, dtype="float32", token_pattern=r"(?u)\b\w+\b")
