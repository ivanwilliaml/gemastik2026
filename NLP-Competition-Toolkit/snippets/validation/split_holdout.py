# title: Holdout stratified dan group
# when: Model mahal, iterasi cepat. Validasi tetap 5-10 ribu baris untuk BERT.
# tags: holdout split
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def holdout(n_or_y, test_size=.2, groups=None, stratify=True, seed=42):
    """Kembalikan (idx_train, idx_valid)."""
    y = np.asarray(n_or_y)
    idx = np.arange(len(y))
    if groups is not None:
        return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, groups=groups))
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=y if stratify else None)
