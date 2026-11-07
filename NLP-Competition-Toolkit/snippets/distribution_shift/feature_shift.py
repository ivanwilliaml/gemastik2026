# title: PSI per fitur dan garis waktu volume
# when: Banyak meta features, atau ada waktu.
# avoid: PSI > 0,25 hanya aturan praktis.
# tags: psi volume garis_waktu lonjakan rezim
import numpy as np
import pandas as pd

def psi(a, b, bins=10, eps=1e-4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(a, edges)[0] / len(a) + eps
    q = np.histogram(b, edges)[0] / len(b) + eps
    return round(float(((q - p) * np.log(q / p)).sum()), 4)

def psi_table(train_df, test_df):
    c = [x for x in train_df.columns if x in test_df.columns and pd.api.types.is_numeric_dtype(train_df[x])]
    return pd.Series({x: psi(train_df[x].dropna(), test_df[x].dropna()) for x in c}).sort_values(ascending=False)

def volume_timeline(ts, freq="D", z=3.0):
    """Volume per periode dan tanda lonjakan. Pakai untuk garis waktu slide."""
    t = pd.to_datetime(pd.Series(list(ts)), errors="coerce").dropna()
    v = t.dt.to_period(freq).value_counts().sort_index()
    med = v.rolling(14, min_periods=3, center=True).median().fillna(v.median())
    mad = (v - med).abs().median() or 1
    return pd.DataFrame({"n": v, "lonjakan": (v - med) / (1.4826 * mad) > z})
