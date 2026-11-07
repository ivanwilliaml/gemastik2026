from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
F = list(StratifiedKFold(3, shuffle=True, random_state=1).split(texts, y))
o, t = oof_predict(TfidfVectorizer(), LogisticRegression(max_iter=500), texts, y, F, texts_test)
assert o.shape == (len(y), 3) and t.shape == (len(texts_test), 3) and abs(t.sum(1).mean() - 1) < 1e-4
o, _ = oof_predict(lambda: TfidfVectorizer(), lambda: LinearSVC(), texts, y, F); assert o.shape[1] == 3 and abs(o.sum(1).mean() - 1) < 1e-4
o, _ = oof_predict(TfidfVectorizer(), Ridge(), texts, y.astype(float), F); assert o.shape[1] == 1
