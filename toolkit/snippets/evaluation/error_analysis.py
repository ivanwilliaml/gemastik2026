# title: Error analysis: yakin-salah, tertukar, segmen, residual
# when: T70-T90. Sumber ide paling jujur.
# avoid: Hanya contoh favorit.
# tags: error confusion segmen residual laporan_per_label
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def worst_errors(texts, y, proba, k=30, names=None):
    y, p = np.asarray(y), np.asarray(proba)
    pred = p.argmax(1)
    d = pd.DataFrame({"text": list(texts), "y": y, "pred": pred, "conf": p.max(1)})[y != pred]
    if names is not None:
        d["y"], d["pred"] = d.y.map(dict(enumerate(names))), d.pred.map(dict(enumerate(names)))
    return d.sort_values("conf", ascending=False).head(k)

def random_errors(texts, y, pred, k=50, seed=42):
    d = pd.DataFrame({"text": list(texts), "y": np.asarray(y), "pred": np.asarray(pred)})
    d = d[d.y != d.pred]
    return d.sample(min(k, len(d)), random_state=seed).assign(kategori="")

def confused_pairs(y, pred, names=None, k=10):
    c = confusion_matrix(y, pred, normalize="true")
    np.fill_diagonal(c, 0)
    o = np.dstack(np.unravel_index(np.argsort(-c.ravel()), c.shape))[0][:k]
    return [((names[a] if names is not None else int(a)), (names[b] if names is not None else int(b)), round(float(c[a, b]), 3)) for a, b in o]

def per_label(y, pred, names=None):
    return pd.DataFrame(classification_report(y, pred, target_names=names, zero_division=0, output_dict=True)).T.round(3)

def segment_scores(y, pred, segments, score=lambda a, b: f1_score(a, b, average="macro", zero_division=0), min_n=30):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pred), "s": np.asarray(segments)})
    r = {k: (len(g), round(score(g.y, g.p), 4)) for k, g in d.groupby("s") if len(g) >= min_n}
    return pd.DataFrame(r, index=["n", "skor"]).T.sort_values("skor")

def residual_table(y, pred, bins=5):
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float)})
    d["err"] = d.p - d.y
    return d.groupby(pd.qcut(d.y, bins, duplicates="drop"), observed=True).err.agg(["count", "mean", "std"]).round(3)
