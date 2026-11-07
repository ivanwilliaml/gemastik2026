w = importance_weights(np.linspace(.1, .9, 100)); assert abs(w.mean() - 1) < 1e-6 and w[-1] > w[0]
