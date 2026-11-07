# title: Temperature scaling, Platt, isotonic
# when: Log loss, atau transformer terlalu yakin. LinearSVC butuh probabilitas.
# avoid: cv='prefit': gagal. Isotonic pada data kecil. AUC: urutan tetap.
# tags: kalibrasi temperature platt isotonic
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

def fit_temperature(y, p, eps=1e-7):
    """p: probabilitas OOF. Kembalikan T; terapkan dengan apply_temperature."""
    logp = np.log(np.clip(np.asarray(p), eps, 1))
    y = np.asarray(y)
    def nll(T):
        q = logp / T
        q = q - np.log(np.exp(q).sum(1, keepdims=True))
        return -q[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(.2, 10), method="bounded").x)

def apply_temperature(p, T, eps=1e-7):
    q = np.exp(np.log(np.clip(np.asarray(p), eps, 1)) / T)
    return q / q.sum(1, keepdims=True)

def calibrate_fitted(model, Xcal, ycal, method="sigmoid"):
    """Kalibrasi model yang sudah dilatih pada data terpisah. method='isotonic' untuk data besar."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(Xcal, ycal)
