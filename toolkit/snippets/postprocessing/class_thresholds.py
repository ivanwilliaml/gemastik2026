# title: Pengali per kelas dan threshold per label
# when: Macro-F1 multikelas, multilabel, ABSA. Nested untuk laporan jujur, setelah blend.
# avoid: OOF kecil. Threshold global 0,5 pada label jarang.
# tags: pengali multiplier threshold per_label macro nested minimal_satu
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def fit_multipliers(y, p, grid=np.linspace(.5, 2.0, 16), rounds=2):
    """Coordinate ascent. Prediksi = argmax(p * w)."""
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(p.shape[1])
    best = f1_score(y, (p * w).argmax(1), average="macro", zero_division=0)
    for _ in range(rounds):
        for k in range(p.shape[1]):
            for g in grid:
                c = w.copy(); c[k] = g
                s = f1_score(y, (p * c).argmax(1), average="macro", zero_division=0)
                if s > best + 1e-5:
                    best, w = s, c
    return w

def nested_multiplier_score(y, p, n=5, seed=42):
    """Kembalikan (macro-F1 argmax, macro-F1 nested, pengali seluruh OOF). Pakai bila nested lebih tinggi melebihi noise."""
    y, p = np.asarray(y), np.asarray(p)
    pred = np.zeros(len(y), int)
    for a, b in StratifiedKFold(n, shuffle=True, random_state=seed).split(p, y):
        pred[b] = (p[b] * fit_multipliers(y[a], p[a])).argmax(1)
    return round(f1_score(y, p.argmax(1), average="macro"), 4), round(f1_score(y, pred, average="macro"), 4), fit_multipliers(y, p)

def fit_label_thresholds(Y, P, grid=np.linspace(.05, .95, 19)):
    Y, P = np.asarray(Y), np.asarray(P)
    return np.array([grid[np.argmax([f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in grid])] for j in range(Y.shape[1])])

def apply_thresholds(P, t, at_least_one=False):
    P = np.asarray(P)
    out = (P >= t).astype(int)
    if at_least_one:
        none = out.sum(1) == 0
        out[none, P[none].argmax(1)] = 1
    return out
