# title: Grafik slide: heatmap, confusion, garis waktu, distribusi
# when: Selagi lomba berjalan. Data tidak boleh keluar, jadi simpan gambar dan masukkan ke slide. Tanpa kaleido.
# avoid: Plotly export tanpa Chrome.
# tags: grafik heatmap confusion slide timeline matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def _save(path):
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_heatmap(table, path="heatmap.png", title="Aspek x sentimen", fmt=".0f", normalize=False):
    t = table.div(table.sum(1), axis=0) if normalize else table
    plt.figure(figsize=(8, max(3, .4 * len(t))))
    sns.heatmap(t, annot=True, fmt=".2f" if normalize else fmt, cmap="YlGnBu", cbar=False)
    plt.title(title)
    return _save(path)

def plot_confusion(y, pred, names=None, path="confusion.png"):
    c = confusion_matrix(y, pred, normalize="true")
    return plot_heatmap(pd.DataFrame(c, index=names, columns=names), path, "Confusion matrix (per baris)", fmt=".2f")

def plot_timeline(ts, path="timeline.png", events=None, freq="W", title="Volume per minggu"):
    """events: {'YYYY-MM-DD': 'nama kejadian'}"""
    t = pd.to_datetime(pd.Series(list(ts)), errors="coerce").dropna()
    v = t.dt.to_period(freq).value_counts().sort_index()
    x = v.index.to_timestamp()
    plt.figure(figsize=(10, 3.6)); plt.plot(x, v.values, color="#0f766e")
    for d, name in (events or {}).items():
        plt.axvline(pd.Timestamp(d), color="#b42318", ls="--", lw=1); plt.text(pd.Timestamp(d), v.max(), name, fontsize=8, rotation=90, va="top")
    plt.title(title)
    return _save(path)

def plot_bars(series, path="bars.png", title="Distribusi", top=20):
    s = pd.Series(series).sort_values().tail(top)
    plt.figure(figsize=(8, max(3, .35 * len(s)))); plt.barh(s.index.astype(str), s.values, color="#0f766e"); plt.title(title)
    return _save(path)
