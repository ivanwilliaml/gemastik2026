import numpy as _n
assert make_rf(5).fit(_n.random.rand(30, 3), _n.arange(30) % 2).predict_proba(_n.random.rand(2, 3)).shape == (2, 2)
