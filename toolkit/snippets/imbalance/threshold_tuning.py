# title: Threshold biner dari kurva OOF dihaluskan
# when: F1 biner. Sering gain terbesar. Pilih puncak yang stabil, nested untuk laporan jujur.
# avoid: AUC atau log loss. Threshold dari model berbeda.
# tags: threshold f1 biner oof nested
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def best_threshold(y, p, grid=None, smooth=5):
    """p: probabilitas kelas positif. Kurva dihaluskan agar tidak mengejar noise."""
    grid = np.linspace(.05, .95, 91) if grid is None else grid
    f = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    return float(grid[np.convolve(f, np.ones(smooth) / smooth, mode="same").argmax()])

def nested_threshold_score(y, p, n=5, seed=42):
    """Skor jujur: threshold difit di fold lain. Kembalikan (F1 nested, threshold seluruh OOF)."""
    y, p = np.asarray(y), np.asarray(p)
    pred = np.zeros(len(y), int)
    for a, b in StratifiedKFold(n, shuffle=True, random_state=seed).split(p, y):
        pred[b] = p[b] >= best_threshold(y[a], p[a])
    return round(f1_score(y, pred, zero_division=0), 4), best_threshold(y, p)
