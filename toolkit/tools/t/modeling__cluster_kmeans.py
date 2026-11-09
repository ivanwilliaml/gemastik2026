from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
Z = TruncatedSVD(16, random_state=1).fit_transform(TfidfVectorizer().fit_transform(texts))
assert len(set(kmeans_l2(Z, 3))) == 3 and search_k(Z, (2, 3), 200)[0]["k"] == 2
assert len(set(consensus_kmeans(Z[:60], 3, (1, 2)))) == 3 and len(agglo_sample(Z[:40], .9)) == 40
