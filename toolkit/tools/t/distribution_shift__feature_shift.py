assert psi(np.arange(100), np.arange(100)) < .01 and psi(np.arange(100), np.arange(100) + 80) > .25
assert psi_table(pd.DataFrame({"a": range(50)}), pd.DataFrame({"a": range(50)})).loc["a"] < .01
v = volume_timeline(pd.date_range("2025-01-01", periods=30).astype(str).tolist() + ["2025-01-15"] * 40); assert v.lonjakan.sum() >= 1
