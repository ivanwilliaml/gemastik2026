# title: HashingVectorizer
# when: 500 ribu baris atau RAM sempit, tanpa menyimpan vocabulary.
# avoid: Butuh interpretasi kata.
# tags: hashing skala
from sklearn.feature_extraction.text import HashingVectorizer

def make_hashing(bits=20, ngram=(1, 2)):
    return HashingVectorizer(n_features=2 ** bits, alternate_sign=False, ngram_range=ngram, dtype="float32")
