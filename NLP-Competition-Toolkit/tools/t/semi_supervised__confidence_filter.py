p_ = np.array([[.95, .05], [.92, .08], [.1, .9], [.55, .45]]); k, l = select_confident(p_, .9, quota=1); assert k.tolist() == [0, 2] and l.tolist() == [0, 1]
assert entropy([[.5, .5]])[0] > entropy([[.9, .1]])[0] and reject_low(p_).tolist() == [0, 0, 1, 0]
