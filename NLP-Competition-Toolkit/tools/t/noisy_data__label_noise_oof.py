p_ = np.eye(3)[y] * .7 + .1; p_[:20] = np.eye(3)[(y[:20] + 1) % 3] * .9 + .0333; assert len(noise_candidates(texts, y, p_, 5)) == 5
assert set(cleanlab_rank(y, p_)[:5]) <= set(range(20))
