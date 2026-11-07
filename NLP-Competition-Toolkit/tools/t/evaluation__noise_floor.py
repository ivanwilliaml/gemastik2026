from sklearn.metrics import accuracy_score as _a
y_ = np.r_[np.zeros(100), np.ones(100)]; p_ = y_.copy(); p_[:30] = 1; q_ = y_.copy(); q_[:5] = 1
s, se, lo, hi = noise_floor(y_, p_, _a, 50); assert s == .85 and se > 0 and lo < hi
s2 = noise_floor(y_, p_, _a, 20, groups=np.arange(200) % 20); assert s2[1] > 0
g = paired_gain(y_, p_, q_, _a, 50); assert g["gain"] > 0 and g["p_b_lebih_baik"] > .9
assert set(learning_curve_oof(lambda i: len(i), 100)) == {.1, .25, .5, 1.0}
