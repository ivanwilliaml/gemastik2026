y_ = np.r_[np.zeros(200), np.ones(50)].astype(int); rng_ = np.random.default_rng(1); p_ = np.clip(y_ * .35 + rng_.random(250) * .5, 0, 1)
t = best_threshold(y_, p_); assert .1 < t < .9 and f1_score(y_, p_ >= t) >= f1_score(y_, p_ >= .5)
assert nested_threshold_score(y_, p_)[0] > 0
