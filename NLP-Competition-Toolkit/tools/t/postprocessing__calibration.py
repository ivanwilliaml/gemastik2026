from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
rng_ = np.random.default_rng(1); y_ = rng_.integers(0, 2, 400); p1 = np.where(rng_.random(400) < .7, y_, 1 - y_); p_ = np.c_[1 - p1, p1] * .98 + .01
T = fit_temperature(y_, p_); assert T > 1 and apply_temperature(p_, T).max() < p_.max()
X = TfidfVectorizer().fit_transform(texts); assert calibrate_fitted(LinearSVC().fit(X[:400], y[:400]), X[400:], y[400:]).predict_proba(X[:2]).shape == (2, 3)
