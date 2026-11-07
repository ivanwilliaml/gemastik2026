# title: Pseudo-label berulang dengan penghentian
# when: Model cepat. Hentikan saat tidak melebihi noise floor pada gold.
# avoid: Transformer berulang: mahal.
# tags: iteratif pseudo stop
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score

def iterative_pseudo(vec, model, tr_t, tr_y, unl_t, gold_t, gold_y, rounds=3, threshold=.95, min_gain=.002):
    """gold: label asli yang tidak pernah dilatih. Kembalikan (vec, model, riwayat)."""
    t, yy, hist, best = list(tr_t), np.asarray(tr_y), [], None
    for r in range(rounds + 1):
        v = clone(vec); m = clone(model)
        m.fit(v.fit_transform(t), yy)
        s = f1_score(gold_y, m.predict(v.transform(gold_t)), average="macro")
        hist.append(round(s, 4))
        if best is not None and s < best[0] + min_gain:
            break
        best = (s, v, m)
        if r == rounds:
            break
        p = m.predict_proba(v.transform(unl_t))
        k = p.max(1) >= threshold
        t, yy = list(tr_t) + [x for x, a in zip(unl_t, k) if a], np.r_[np.asarray(tr_y), m.classes_[p[k].argmax(1)]]
    return best[1], best[2], hist
