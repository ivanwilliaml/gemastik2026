# title: Pseudo-label satu putaran
# when: Banyak unlabeled dan aturan mengizinkan. Evaluasi hanya pada label asli.
# avoid: Pseudo-label masuk validasi.
# tags: pseudo semi
import numpy as np
from sklearn.base import clone

def pseudo_fit(vec, model, texts, y, unl_texts, threshold=.95):
    """Fit, ambil baris yakin, fit ulang. Kembalikan (vec, model, jumlah pseudo)."""
    v = clone(vec); m = clone(model)
    m.fit(v.fit_transform(texts), y)
    p = m.predict_proba(v.transform(unl_texts))
    keep = p.max(1) >= threshold
    t2 = list(texts) + [t for t, k in zip(unl_texts, keep) if k]
    y2 = np.r_[np.asarray(y), m.classes_[p[keep].argmax(1)]]
    v2 = clone(vec); m2 = clone(model)
    m2.fit(v2.fit_transform(t2), y2)
    return v2, m2, int(keep.sum())
