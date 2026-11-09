# title: Tren isu: proporsi, titik perubahan, proporsi topik
# when: Ada kolom waktu. Laporkan proporsi, bukan hanya jumlah: kenaikan keluhan belum tentu kenaikan porsi.
# avoid: Periode dengan volume kecil: proporsinya melonjak karena pembagi kecil.
# tags: tren waktu temporal lonjakan change point ruptures proporsi timeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

def _period(times, freq):
    t = pd.to_datetime(pd.Series(list(times)), errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    return t, t.dt.to_period(freq)

def issue_timeline(times, mask=None, freq="W", min_total=30):
    """Tabel per periode: total, isu, proporsi. Periode di bawah min_total ditandai agar tidak dibaca."""
    t, p = _period(times, freq)
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)
    tl = pd.DataFrame({"total": p[t.notna()].value_counts(), "isu": p[t.notna() & m].value_counts()}).fillna(0).astype(int).sort_index()
    tl["proporsi"] = (tl.isu / tl.total).round(4)
    tl["cukup"] = tl.total >= min_total
    return tl

def change_points(tl, col="proporsi", penalty=None, min_size=2):
    """Titik perubahan pada periode yang cukup datanya. Kembalikan daftar periode awal segmen baru."""
    s = tl[tl.cukup][col] if "cukup" in tl else tl[col]
    if len(s) < 2 * min_size + 1:
        return []
    x = s.to_numpy(float).reshape(-1, 1)
    pen = penalty if penalty is not None else 3 * np.log(len(x)) * max(float(np.var(x)), 1e-9)
    bk = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(x).predict(pen=pen)
    return [s.index[i] for i in bk[:-1]]

def share_over_time(times, labels, freq="M", min_total=30):
    """Proporsi tiap topik per periode. Periode kecil dibuang."""
    t, p = _period(times, freq)
    ok = t.notna().to_numpy()
    c = pd.crosstab(p[ok].to_numpy(), np.asarray(labels)[ok])
    c = c[c.sum(1) >= min_total]
    return c.div(c.sum(1), axis=0).round(3)

def plot_issue_timeline(tl, path="tren_isu.png", title="Porsi isu per periode", events=None, cps=None):
    d = tl[tl.cukup] if "cukup" in tl else tl
    x = d.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x, d.total, width=(x[1] - x[0]) * .8 if len(x) > 1 else 1, color="#d0d5dd")
    ax.set_ylabel("volume")
    a2 = ax.twinx()
    a2.plot(x, d.proporsi, color="#b42318", lw=2)
    a2.set_ylabel("porsi isu")
    for c in cps or []:
        ax.axvline(c.to_timestamp(), color="#0f766e", lw=1.5)
    for when, name in (events or {}).items():
        w = pd.Timestamp(when)
        if len(x) and x[0] <= w <= x[-1]:
            ax.axvline(w, color="#667085", ls="--", lw=1)
            ax.text(w, d.total.max(), name, fontsize=8, rotation=90, va="top")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def bertopic_over_time(docs, times, embeddings, nr_bins=12, min_topic_size=30):
    """BERTopic pada embedding yang sudah ada. Pakai sampel 20-50 ribu baris; RAM 500 ribu belum teruji."""
    from bertopic import BERTopic
    tm = BERTopic(min_topic_size=min_topic_size, calculate_probabilities=False, low_memory=True, verbose=False)
    tm.fit(docs, embeddings)
    return tm, tm.topics_over_time(docs, list(pd.to_datetime(pd.Series(list(times)), errors="coerce", utc=True, format="mixed").dt.tz_localize(None)), nr_bins=nr_bins)
