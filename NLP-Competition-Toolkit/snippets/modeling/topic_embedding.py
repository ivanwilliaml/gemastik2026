# title: BERTopic pada sampel dan fallback MiniBatch
# when: Dedup, embedding sekali, UMAP + HDBSCAN pada 20-50 ribu, lalu assign sisanya.
# avoid: Seluruh 500 ribu baris tanpa uji RAM. calculate_probabilities=True.
# tags: bertopic topik umap hdbscan fallback assign
# needs: modeling/keyword_extract
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize

def bertopic_sample(docs, E, sample=30000, min_topic_size=50, seed=42):
    """Fit pada sampel, tetapkan sisanya ke pusat topik terdekat. Kembalikan (label per baris, model)."""
    from bertopic import BERTopic
    import hdbscan, umap
    from sklearn.feature_extraction.text import CountVectorizer
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(docs), min(sample, len(docs)), replace=False)
    docs_s = [str(docs[i]) for i in idx]
    Es = np.asarray(E[idx])
    m = BERTopic(umap_model=umap.UMAP(n_components=5, n_neighbors=15, metric="cosine", low_memory=True, random_state=seed),
                 hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=min_topic_size, prediction_data=False),
                 vectorizer_model=CountVectorizer(min_df=2, token_pattern=r"(?u)\b[^\d\W]{2,}\b"),
                 low_memory=True, calculate_probabilities=False, verbose=False)
    t, _ = m.fit_transform(docs_s, embeddings=Es)
    t = np.asarray(t)
    if (t >= 0).sum() == 0 or len(set(t[t >= 0])) < 2:
        return np.where(np.zeros(len(docs), bool), 0, -1), m
    nc = NearestCentroid(metric="euclidean").fit(Es[t >= 0], t[t >= 0])
    return nc.predict(np.asarray(E)), m

def kmeans_topics(docs, E, k=30, top=10, pca=50, seed=42):
    """Fallback: IncrementalPCA + MiniBatchKMeans + c-TF-IDF. Kembalikan (label, kata per topik)."""
    Z = IncrementalPCA(min(pca, np.asarray(E).shape[1]), batch_size=max(4096, pca * 2)).fit_transform(E) if np.asarray(E).shape[1] > pca else np.asarray(E)
    lab = MiniBatchKMeans(k, batch_size=4096, n_init=5, random_state=seed).fit_predict(normalize(Z))
    return lab, ctfidf_terms(docs, lab, top)
