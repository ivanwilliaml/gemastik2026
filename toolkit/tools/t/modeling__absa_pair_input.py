L_ = absa_long(_pd_absa.head(5), _pd_absa.columns[0], _pd_absa.columns[1:3]); assert len(L_) == 10 and {"text", "aspect", "label"} <= set(L_.columns)
w = absa_wide(L_, ["x"] * 10, np.tile(np.arange(5), 2)); assert w.shape == (5, 2)
