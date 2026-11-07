# title: Length shift dan uji KS
# when: Sebelum memakai fitur panjang.
# avoid: p-value pada data besar selalu kecil. Lihat statistik D.
# tags: panjang ks
import numpy as np
from scipy.stats import ks_2samp

def length_shift(train, test):
    a, b = [np.array([len(str(t).split()) for t in x]) for x in (train, test)]
    r = ks_2samp(a, b)
    return dict(d=round(float(r.statistic), 4), p=round(float(r.pvalue), 4), p50=(float(np.median(a)), float(np.median(b))), p95=(float(np.quantile(a, .95)), float(np.quantile(b, .95))))
