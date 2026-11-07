# title: Importance weighting
# when: Covariate shift ringan sampai sedang. Clip dan cek effective sample size.
# avoid: Bobot ekstrem: varian naik.
# tags: importance bobot covariate ess
import numpy as np

def importance_weights(p_test_on_train, ratio=1.0, clip=(.01, .99)):
    """p: probabilitas 'test' untuk baris train dari adversarial. ratio = n_train/n_test."""
    p = np.clip(np.asarray(p_test_on_train), *clip)
    w = ratio * p / (1 - p)
    lo, hi = np.quantile(w, [.01, .99])
    w = np.clip(w, lo, hi)
    w = w / w.mean()
    print("ESS", int(w.sum() ** 2 / (w ** 2).sum()), "dari", len(w))
    return w
