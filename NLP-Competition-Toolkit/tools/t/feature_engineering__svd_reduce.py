from sklearn.feature_extraction.text import TfidfVectorizer
Z = make_lsa(TfidfVectorizer(), k=16).fit_transform(texts); assert Z.shape[1] == 16
assert ipca(Z, 4, 64).shape[1] == 4 and tsne_2d(Z[:40]).shape == (40, 2) and umap_reduce(Z[:80], 2, 5).shape == (80, 2)
