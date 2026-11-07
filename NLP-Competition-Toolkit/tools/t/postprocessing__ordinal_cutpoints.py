rng_ = np.random.default_rng(1); y_ = rng_.integers(1, 6, 300); p_ = y_ * .4 + rng_.normal(0, .3, 300)
c = fit_cuts(y_, p_); assert len(c) == 4 and cohen_kappa_score(y_, apply_cuts(p_, c, 1), weights="quadratic") > .6 and nested_cuts_score(y_, p_, 3) > .5
assert to_output(np.log1p([9.]), log=True, lo=0, hi=5, integer=True).tolist() == [5]
