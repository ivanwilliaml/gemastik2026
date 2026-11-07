y_ = np.r_[np.zeros(200), np.ones(30), np.full(30, 2)].astype(int); rng_ = np.random.default_rng(1)
p_ = rng_.dirichlet((3, 1, 1), 260); p_[np.arange(260), y_] += .25; p_ /= p_.sum(1, keepdims=True)
w = fit_multipliers(y_, p_); assert f1_score(y_, (p_ * w).argmax(1), average="macro") >= f1_score(y_, p_.argmax(1), average="macro")
assert nested_multiplier_score(y_, p_, 3)[1] > 0
P_ = rng_.random((100, 3)); Y_ = (P_ > .7).astype(int); t = fit_label_thresholds(Y_, P_); assert (.5 < t).all() and apply_thresholds(np.array([[.1, .2]]), np.array([.5, .5]), True).tolist() == [[0, 1]]
