# title: Cut point QWK, inverse log1p, dan clip
# when: Regresi ordinal: DataQuest +0,055. Nested. Jangan refit setelah cut ditetapkan.
# avoid: Metrik pada skala log tanpa sengaja.
# tags: qwk cutpoint ordinal clip log1p
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

def apply_cuts(p, cuts, low=0):
    return low + np.digitize(np.asarray(p), np.sort(cuts))

def fit_cuts(y, p, low=None, rounds=2, n_grid=31):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    low = int(y.min()) if low is None else low
    k = int(y.max()) - low
    cuts = np.quantile(p, np.cumsum(np.bincount(y - low))[:-1] / len(y))
    score = lambda c: cohen_kappa_score(y, apply_cuts(p, c, low), weights="quadratic")
    best = score(cuts)
    for _ in range(rounds):
        for i in range(k):
            for g in np.linspace(cuts[i] - .5 * p.std(), cuts[i] + .5 * p.std(), n_grid):
                c = cuts.copy(); c[i] = g
                s = score(c)
                if s > best + 1e-5:
                    best, cuts = s, c
    return np.sort(cuts)

def nested_cuts_score(y, p, n=5, seed=42):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pred = np.zeros(len(y), int)
    low = int(y.min())
    for a, b in KFold(n, shuffle=True, random_state=seed).split(p):
        pred[b] = apply_cuts(p[b], fit_cuts(y[a], p[a], low=low), low)
    return round(cohen_kappa_score(y, pred, weights="quadratic"), 4)

def to_output(p, log=False, lo=None, hi=None, integer=False):
    p = np.expm1(p) if log else np.asarray(p)
    p = np.clip(p, lo, hi) if lo is not None or hi is not None else p
    return np.rint(p).astype(int) if integer else p
