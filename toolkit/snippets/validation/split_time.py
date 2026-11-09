# title: Time split dan expanding folds
# when: Test dari periode lebih baru. Fit TF-IDF hanya pada masa lampau.
# avoid: Random split: kebocoran masa depan.
# tags: waktu time temporal
import numpy as np
import pandas as pd

def time_holdout(ts, valid_share=.2, gap_days=0):
    t = pd.to_datetime(pd.Series(list(ts)), errors="coerce")
    cut = t.quantile(1 - valid_share)
    tr = np.where(t < cut - pd.Timedelta(days=gap_days))[0]
    return tr, np.where(t >= cut)[0]

def time_folds(ts, n=3):
    t = pd.to_datetime(pd.Series(list(ts)), errors="coerce")
    q = t.quantile(np.linspace(.5, 1, n + 1))
    return [(np.where(t < q.iloc[i])[0], np.where((t >= q.iloc[i]) & (t < q.iloc[i + 1] if i < n - 1 else t <= q.iloc[i + 1]))[0]) for i in range(n)]
