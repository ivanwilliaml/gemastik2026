# title: LSA, PCA, UMAP, t-SNE
# when: Sebelum tree atau klaster, dan peta slide. Pipeline LSA bisa langsung jadi vectorizer.
# avoid: UMAP dan t-SNE pada semua baris. PCA biasa pada sparse.
# tags: svd lsa pca umap tsne reduksi
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def make_lsa(base_vectorizer, k=256, seed=42):
    """make_lsa(make_tfidf_word()) adalah vectorizer padat ter-L2."""
    return make_pipeline(base_vectorizer, TruncatedSVD(k, random_state=seed), Normalizer(copy=False))

def ipca(E, k=50, batch=4096):
    return IncrementalPCA(k, batch_size=batch).fit_transform(E)

def umap_reduce(E, k=5, neighbors=15, seed=42):
    import umap
    return umap.UMAP(n_components=k, n_neighbors=neighbors, metric="cosine", low_memory=True, random_state=seed).fit_transform(E)

def tsne_2d(E, seed=42):
    return TSNE(2, metric="cosine", init="pca", random_state=seed, perplexity=min(30, len(E) // 4)).fit_transform(E)
