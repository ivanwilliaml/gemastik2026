# title: Metrik regresi, QWK, RMSLE, korelasi
# when: Regresi dan similarity.
# avoid: squared=False: sudah hilang.
# tags: rmse mae r2 mape qwk rmsle pearson spearman
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, r2_score, root_mean_squared_error

def reg_report(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    r = dict(rmse=root_mean_squared_error(y, pred), mae=mean_absolute_error(y, pred), r2=r2_score(y, pred), pearson=pearsonr(y, pred)[0], spearman=spearmanr(y, pred)[0])
    if (y > 0).all():
        r["mape"] = float(np.mean(np.abs((y - pred) / y)))
    if (y >= 0).all() and (pred >= 0).all():
        r["rmsle"] = root_mean_squared_error(np.log1p(y), np.log1p(pred))
    return {k: round(float(v), 4) for k, v in r.items()}

def qwk(y, pred):
    return float(cohen_kappa_score(np.asarray(y).astype(int), np.asarray(pred).astype(int), weights="quadratic"))
