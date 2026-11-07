# title: Stratified K-Fold dan K-Fold
# when: Klasifikasi i.i.d. Regresi: target di-bin agar tiap fold mewakili rentang target.
# avoid: Grup tersembunyi lintas fold.
# tags: stratified kfold regresi
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def folds_stratified(y, n=5, seed=42):
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

def folds_regression(y, n=5, bins=10, seed=42):
    y = np.asarray(y, dtype=float)
    q = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]))
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(y, np.digitize(y, q)))

def folds_random(n_rows, n=5, seed=42):
    return list(KFold(n, shuffle=True, random_state=seed).split(np.zeros(n_rows)))
