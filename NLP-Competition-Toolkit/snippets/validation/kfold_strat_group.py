# title: Stratified Group K-Fold
# when: Duplikat dan imbalance bersamaan, umum di media sosial.
# tags: stratified group duplikat
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def folds_strat_group(y, groups, n=5, seed=42):
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
