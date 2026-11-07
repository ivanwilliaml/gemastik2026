d = pd.DataFrame({"a": [1, 2] * 50, "b": ["x", "y"] * 50, "c": [.5, 1.5] * 50})
assert reduce_memory(d).memory_usage(deep=True).sum() < d.memory_usage(deep=True).sum()
