# title: Distribusi label, multilabel, dan target numerik
# when: Sebelum memilih metrik dan fold.
# tags: imbalance label distribusi cooccurrence skew
import numpy as np
import pandas as pd

def label_table(y):
    s = pd.Series(np.asarray(y)).value_counts()
    print("rasio mayor:minor", round(s.max() / s.min(), 1))
    return pd.DataFrame({"n": s, "pct": (s / s.sum()).round(4)})

def multilabel_table(Y, names=None):
    Y = np.asarray(Y)
    names = names if names is not None else [f"l{i}" for i in range(Y.shape[1])]
    print("rata-rata label per baris", round(Y.sum(1).mean(), 3), "| baris tanpa label", round((Y.sum(1) == 0).mean(), 3))
    co = pd.DataFrame(Y.T @ Y, index=names, columns=names)
    return pd.DataFrame({"n": Y.sum(0), "pct": Y.mean(0).round(4)}, index=names), co

def target_table(y):
    s = pd.Series(np.asarray(y, dtype=float))
    print("skew", round(s.skew(), 3), "| nol", round((s == 0).mean(), 3), "| unik", s.nunique(), "| disarankan log1p" if s.skew() > 1 and s.min() >= 0 else "")
    return s.describe(percentiles=[.01, .5, .99]).round(3)
