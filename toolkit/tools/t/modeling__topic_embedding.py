from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer as _T
Z = normalize(TruncatedSVD(32, random_state=1).fit_transform(_T().fit_transform(texts)))
l, words = kmeans_topics(texts, Z, k=4, top=5, pca=16); assert len(l) == len(texts) and len(words) == 4
l, m = bertopic_sample(texts, Z, min_topic_size=10); assert len(l) == len(texts)
