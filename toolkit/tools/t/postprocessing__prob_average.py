assert np.allclose(blend([np.array([[.2, .8]]), np.array([[.4, .6]])]), [[.3, .7]])
assert blend([np.array([1.]), np.array([3.])], [3, 1])[0] == 1.5
