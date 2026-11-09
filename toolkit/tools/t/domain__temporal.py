import pandas as pd, numpy as np, tempfile, os
times = pd.date_range("2025-01-01", periods=2000, freq="6h").astype(str).tolist()
mask = np.r_[np.random.default_rng(0).random(1000) < .02, np.random.default_rng(1).random(1000) < .5]
tl = issue_timeline(times, mask, "W", min_total=20)
assert tl.total.sum() == 2000 and tl.proporsi.iloc[-2] > tl.proporsi.iloc[1]
cp = change_points(tl)
assert len(cp) >= 1, cp
assert abs((cp[0].start_time - pd.Timestamp(times[1000])).days) <= 14
assert os.path.exists(plot_issue_timeline(tl, os.path.join(tempfile.mkdtemp(), "t.png"), events={"2025-06-01": "x"}, cps=cp))
assert share_over_time(times, mask.astype(int), "M").sum(axis=1).round(2).eq(1).all()
assert callable(bertopic_over_time)
