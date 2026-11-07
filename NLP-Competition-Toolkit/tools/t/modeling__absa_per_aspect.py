A_ = _pd_absa.head(150)
o, t = absa_cv(A_.iloc[:, 0].tolist(), A_.iloc[:, 1:], A_.iloc[:5, 0].tolist(), n=3); assert set(o) == set(A_.columns[1:]) and len(t[A_.columns[1]]) == 5
