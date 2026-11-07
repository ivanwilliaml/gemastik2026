import numpy as _n
X_ = _n.random.default_rng(1).random((60, 3)); y_ = _n.r_[_n.zeros(50), _n.ones(10)].astype(int); assert make_smote_lr(3).fit(X_, y_).predict_proba(X_).shape == (60, 2)
