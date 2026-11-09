# title: Iterative stratification tanpa paket tambahan
# when: Multilabel dan ABSA: menjaga proporsi label jarang.
# tags: multilabel stratification
import numpy as np

def folds_multilabel(Y, n=5, seed=42):
    """Sechidis dkk. Kembalikan list (idx_train, idx_valid)."""
    raw = np.asarray(Y, dtype=float)
    assert np.isin(raw[np.isfinite(raw)], [0, 1]).all(), "Target harus 0/1/NaN"
    Y = np.hstack([raw == 1, np.isfinite(raw)])
    rng = np.random.default_rng(seed)
    left = np.ones(len(Y), bool)
    fold = -np.ones(len(Y), int)
    need = np.tile(Y.sum(0) / n, (n, 1)).astype(float)
    cap = np.full(n, len(Y) / n)
    while left.any():
        counts = Y[left].sum(0)
        counts[counts == 0] = 10 ** 9
        lab = counts.argmin()
        if counts[lab] == 10 ** 9:
            for i in np.where(left)[0]:
                f = int(cap.argmax())
                fold[i] = f; cap[f] -= 1; left[i] = False
            break
        for i in rng.permutation(np.where(left & Y[:, lab])[0]):
            d = need[:, lab]
            cand = np.where(d == d.max())[0]
            f = int(cand[cap[cand].argmax()])
            fold[i] = f; left[i] = False; cap[f] -= 1
            need[f, Y[i]] -= 1
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n)]
