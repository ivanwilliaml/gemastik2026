rng_ = np.random.default_rng(1); y_ = np.arange(100) % 2; a = np.c_[1 - y_ * .8 - .1, y_ * .8 + .1]
o, t = stack_cls([a, a], y_, [a[:3], a[:3]]); assert o.shape == (100, 2) and t.shape == (3, 2)
o, t = stack_reg([y_ * 1., y_ * 2.], y_ * 1., [np.ones(2), np.ones(2)]); assert o.shape == (100,) and t.shape == (2,)
