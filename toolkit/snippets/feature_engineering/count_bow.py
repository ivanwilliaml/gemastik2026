# title: Bag of Words, CountVectorizer, Binary BoW
# when: Naive Bayes, LDA, atau baseline sangat cepat. Semua make_* mengembalikan vectorizer gaya scikit-learn.
# tags: bow count binary
from sklearn.feature_extraction.text import CountVectorizer

def make_count(ngram=(1, 1), min_df=2, max_df=.95, max_features=None, binary=False):
    return CountVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df, max_features=max_features, binary=binary)
