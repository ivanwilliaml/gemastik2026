from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
v, m, h = iterative_pseudo(TfidfVectorizer(), LogisticRegression(C=50, max_iter=500), texts[:200], y[:200], texts[200:450], texts[450:], y[450:], 2, .7); assert 1 <= len(h) <= 3
