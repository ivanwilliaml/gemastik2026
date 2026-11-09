# title: Branch B: tokenisasi, stopword, stemming
# when: Topik, klaster, dan kata kunci. Stem kosakata unik: 36 ribu kata sekitar 4 detik.
# avoid: Classifier: buang stopword menurunkan skor pada uji lokal.
# tags: stemming sastrawi stopword tokenisasi lemmatization
import re

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(text).lower())

def stem_corpus(docs, stop=()):
    """Stem kata unik lalu petakan, jauh lebih cepat daripada per dokumen."""
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer, stop = StemmerFactory().create_stemmer(), set(stop)
    toks = [[w for w in tokenize(d) if w not in stop] for d in docs]
    vocab = {w for t in toks for w in t}
    m = {w: stemmer.stem(w) for w in vocab}
    return [" ".join(m[w] for w in t) for t in toks]
