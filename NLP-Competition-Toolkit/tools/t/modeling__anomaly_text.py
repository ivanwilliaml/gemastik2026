from sklearn.feature_extraction.text import TfidfVectorizer as _T
from sklearn.decomposition import TruncatedSVD as _S
X = _T().fit_transform(texts); Z = _S(16, random_state=1).fit_transform(X)
s = rank_avg(iforest_score(Z), knn_score(Z, 5), lof_score(Z, 5), recon_score(X, 8)); assert len(s) == len(texts)
r = inject_eval(texts, lambda t: recon_score(_T().fit_transform(t), 8), ["zzz qqq xxx www", "lorem ipsum dolor sit amet"], share=.03); assert 0 <= r["auc"] <= 1
