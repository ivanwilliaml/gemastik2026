# title: Filter keyakinan dengan kuota per kelas
# when: Selalu bersama pseudo-label. Mencegah kelas mudah mendominasi.
# avoid: Satu threshold global tanpa kuota.
# tags: confidence kuota margin entropi out_of_scope
import numpy as np

def select_confident(p, threshold=.9, quota=None, margin=None):
    """Kembalikan indeks terpilih dan label. quota: maks per kelas."""
    p = np.asarray(p)
    lab, conf = p.argmax(1), p.max(1)
    ok = conf >= threshold
    if margin is not None:
        ok &= (np.sort(p, 1)[:, -1] - np.sort(p, 1)[:, -2]) >= margin
    keep = []
    for k in range(p.shape[1]):
        ix = np.where(ok & (lab == k))[0]
        ix = ix[np.argsort(-conf[ix])]
        keep.append(ix[:quota] if quota else ix)
    keep = np.sort(np.concatenate(keep))
    return keep, lab[keep]

def entropy(p, eps=1e-12):
    p = np.asarray(p)
    return -(p * np.log(p + eps)).sum(1)

def reject_low(p, threshold=.5, other=-1):
    """Beri kelas 'lainnya' bila keyakinan rendah (out-of-scope)."""
    return np.where(np.asarray(p).max(1) >= threshold, np.asarray(p).argmax(1), other)
