# title: Ridge, LightGBM regresi, baseline, dan log1p
# when: Regresi teks. Objective sesuai metrik: l1 untuk MAE, l2 untuk RMSE. Target berekor panjang: log=True.
# avoid: Target skew tanpa log1p.
# tags: regresi ridge log1p dummy
import lightgbm as lgb
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

def _maybe_log(m, log):
    return TransformedTargetRegressor(m, func=np.log1p, inverse_func=np.expm1) if log else m

def make_ridge(alpha=1.0, log=False):
    return _maybe_log(Ridge(alpha=alpha, solver="sag", random_state=42), log)

def make_lgbm_reg(objective="l2", n=500, lr=.05, log=False, n_jobs=8):
    return _maybe_log(lgb.LGBMRegressor(objective=objective, n_estimators=n, learning_rate=lr, verbose=-1, n_jobs=n_jobs), log)

def make_dummy_reg(strategy="mean"):
    return DummyRegressor(strategy=strategy)
