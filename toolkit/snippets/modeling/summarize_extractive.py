# title: Ringkasan: Lead-N, TextRank, Kalimat
# when: Ekstraktif tanpa model. Cek ada tanda baca dulu.
# avoid: TextRank untuk dokumen satu kalimat.
# tags: ringkasan textrank lead pisah_kalimat
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

def split_sents(text, window=None):
    """Tanpa tanda baca, gunakan window=30 untuk potongan kata."""
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(text)) if x.strip()]
    if window and len(s) <= 1:
        w = str(text).split()
        return [" ".join(w[i:i + window]) for i in range(0, len(w), window)]
    return s

def lead_n(text, n=3):
    return " ".join(split_sents(text)[:n])

def textrank(text, n=3):
    s = split_sents(text)
    if len(s) <= n:
        return " ".join(s)
    X = TfidfVectorizer().fit_transform(s)
    S = (X @ X.T).tolil()
    S.setdiag(0)                    # tanpa self-loop: kalimat terisolasi tidak boleh menang
    r = nx.pagerank(nx.from_scipy_sparse_array(S.tocsr()), max_iter=200)
    keep = sorted(sorted(r, key=r.get, reverse=True)[:n])
    return " ".join(s[i] for i in keep)
