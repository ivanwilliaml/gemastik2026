ts_ = pd.date_range("2025-01-01", periods=100).astype(str).tolist()
a, b = time_holdout(ts_, gap_days=2); assert a.max() < b.min() and b.min() - a.max() >= 2
assert all(a.max() < b.min() for a, b in time_folds(ts_))
