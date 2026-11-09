# title: Rata-rata rank
# when: Biner, AUC, ranking. Tahan beda skala.
# avoid: Log loss.
# tags: rank blend hybrid
import numpy as np
from scipy.stats import rankdata

def rank_blend(preds, weights=None):
    return np.average([rankdata(p) / len(p) for p in preds], axis=0, weights=weights)
