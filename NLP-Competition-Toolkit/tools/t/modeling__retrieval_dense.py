D = np.eye(4); i, s = topk_dense(np.eye(4)[:2], D, 2); assert i[0, 0] == 0 and i[1, 0] == 1
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer().fit(texts); i, s = topk_sparse(v.transform(texts[:3]), v.transform(texts), 2); assert (i[:, 0] == np.arange(3)).all()
assert rrf_merge([np.array([[1, 2]]), np.array([[1, 3]])], top=1).tolist() == [[1]]
