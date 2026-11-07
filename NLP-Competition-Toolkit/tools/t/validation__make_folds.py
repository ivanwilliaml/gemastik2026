from sklearn.feature_extraction.text import TfidfVectorizer as _T
from sklearn.linear_model import LogisticRegression as _L, Ridge as _R
assert len(make_folds(y, n=3)) == 3 and len(make_folds(y, np.arange(len(y)) % 20, n=3)) == 3 and len(make_folds(Y, n=3)) == 3 and len(make_folds(np.arange(100.), n=3)) == 3 and len(make_folds(y, kind="kfold", n=2)) == 2
ts_ = pd.date_range("2025-01-01", periods=50).astype(str); a, b = make_folds(times=ts_)[0]; assert a.max() < b.min()
assert len(make_folds(groups=np.arange(12) % 3, kind="source")) == 3 and check_folds(["a", "a", "b", "c"], [(np.array([0, 2]), np.array([1, 3]))]) == [.5]
f = make_folds(y, n=3); o, t = oof_predict(_T, lambda: _L(max_iter=300), texts, y, f, texts_test); assert o.shape == (len(y), 3) and t.shape[0] == len(texts_test) and not np.isnan(o).any()
o, _ = oof_predict(_T, _R, texts, y.astype(float), f); assert o.shape[1] == 1
