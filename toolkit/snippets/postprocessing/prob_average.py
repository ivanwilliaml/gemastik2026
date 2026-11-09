# title: Rata-rata probabilitas dan regresi
# when: Multikelas: linear + transformer. Bobot sama lebih dulu. Terima bila gain > 2 SE.
# avoid: Bobot difit pada kandidat hampir seri.
# tags: blend ensemble rata_rata seed_bagging
import numpy as np

def blend(preds, weights=None):
    """preds: list array sama bentuk. Bekerja untuk probabilitas, nilai regresi, dan bagging seed."""
    return np.average(np.stack(preds), axis=0, weights=weights)
