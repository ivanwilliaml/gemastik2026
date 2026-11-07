# title: Subsample stratified
# when: 10-20% untuk iterasi, atau 50-100 ribu untuk BERT.
# avoid: Keputusan akhir dari dry-run.
# tags: sampel stratified subset
import numpy as np
from sklearn.model_selection import train_test_split

def subsample_idx(y, n, seed=42, groups=None):
    """Indeks n baris, proporsi kelas terjaga."""
    y = np.asarray(y)
    if n >= len(y):
        return np.arange(len(y))
    _, c = np.unique(y, return_counts=True)
    strat = y if c.min() >= 2 else None
    return train_test_split(np.arange(len(y)), train_size=n, random_state=seed, stratify=strat)[0]
