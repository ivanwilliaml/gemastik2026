from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
v, m, n = pseudo_fit(TfidfVectorizer(), LogisticRegression(C=50, max_iter=500), texts[:300], y[:300], texts[300:], .6); assert n > 0 and m.predict(v.transform(texts_test)).shape[0] == len(texts_test)
