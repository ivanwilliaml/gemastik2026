import os
assert os.path.exists(plot_heatmap(pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"]), normalize=True))
assert os.path.exists(plot_confusion([0, 1, 1], [0, 1, 0], ["n", "p"])) and os.path.exists(plot_bars({"a": 1, "b": 2}))
assert os.path.exists(plot_timeline(pd.date_range("2025-01-01", periods=60).astype(str).tolist(), events={"2025-02-01": "KLB"}))
