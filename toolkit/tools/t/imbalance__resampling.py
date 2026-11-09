y_ = np.r_[np.zeros(100), np.ones(10)].astype(int); i_ = np.arange(110)
assert (y_[undersample(i_, y_, 2)] == 0).sum() == 20 and (y_[oversample(i_, y_)] == 1).sum() == 100
assert abs(correct_prior([[.5, .5]], [.5, .5], [.9, .1])[0, 0] - .9) < 1e-6
