# title: Lacak sumber label: model, leksikon, atau aturan
# when: Audit asal label pada sampel. Jumlah baris dan agreement tinggi tidak membuktikan pembuat label.
# avoid: Memakai hasilnya diam-diam. Cek aturan pretrained, lalu sampaikan terbuka di presentasi.
# tags: label sumber sidik jari fingerprint model leksikon inset noise weak mesin
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def _agreement(pred, labels):
    """Cocokkan nama kelas lewat mayoritas, lalu hitung persetujuan. Aman walau label berupa angka atau bahasa lain."""
    p, y = pd.Series(np.asarray(pred).astype(str)), pd.Series(np.asarray(labels).astype(str))
    mapping = pd.crosstab(p, y).idxmax(axis=1)
    m = p.map(mapping)
    return float((m == y).mean()), float(cohen_kappa_score(m, y)), mapping.to_dict()

def _light_clean(t):
    t = re.sub(r"https?://\S+|@\w+|#", " ", str(t).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()

def _predict_model(path, texts, batch=64, max_len=128):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    order, out = np.argsort([len(t) for t in texts]), np.zeros(len(texts), dtype=int)
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            ix = order[i:i + batch]
            e = tok([texts[j] for j in ix], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            out[ix] = model(**e).logits.argmax(1).cpu().numpy()
    del model
    return out

def _inset_score(texts, folder):
    """Skor InSet: jumlah bobot per kata, lalu tanda."""
    w = {}
    for f in ("positive.tsv", "negative.tsv"):
        d = pd.read_csv(Path(folder) / f, sep="\t")
        w.update(dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1])))
    s = [sum(w.get(x, 0) for x in t.split()) for t in texts]
    return np.sign(s).astype(int)

def label_fingerprint(texts, labels, models_dir=None, names=("sent-mdhugol", "sent-w11wo", "sent-crypter70", "sent-aardiiiiy"), inset_dir=None, n=1500, seed=42):
    """Bandingkan label dengan calon sumber, pada teks mentah dan teks bersih. Kembalikan tabel terurut."""
    d = pd.DataFrame({"t": pd.Series(list(texts)).astype(str).to_numpy(), "y": list(labels)}).dropna()
    d = d.drop_duplicates("t").sample(min(n, len(d)), random_state=seed)
    views = {"mentah": d.t.tolist(), "bersih": [_light_clean(t) for t in d.t]}
    rows = []
    for name in names:
        p = Path(models_dir or "") / name
        if not p.exists():
            continue
        preds = {}
        for vname, v in views.items():
            preds[vname] = _predict_model(p, v)
            a, k, m = _agreement(preds[vname], d.y)
            rows.append({"sumber": name, "teks": vname, "setuju": round(a, 4), "kappa": round(k, 3)})
    if inset_dir and Path(inset_dir).exists():
        for vname, v in views.items():
            a, k, _ = _agreement(_inset_score(v, inset_dir), d.y)
            rows.append({"sumber": "InSet", "teks": vname, "setuju": round(a, 4), "kappa": round(k, 3)})
    base = float(d.y.astype(str).value_counts(normalize=True).iloc[0])
    r = pd.DataFrame(rows, columns=["sumber", "teks", "setuju", "kappa"]).sort_values("setuju", ascending=False).reset_index(drop=True)
    top = r.iloc[0] if len(r) else None
    if top is None:
        verdict = "Tidak ada calon yang bisa diuji."
    elif top.setuju >= .98:
        verdict = f"Agreement tinggi dengan {top.sumber}; belum membuktikan sumber label. Audit baseline mayoritas, mapping dan gold independen."
    elif top.setuju >= .85:
        verdict = f"Label dekat dengan {top.sumber}: mungkin turunan, voting, atau preprocessing lain. Coba varian."
    else:
        verdict = "Tidak ada sumber yang terbukti. Label belum tentu manual: sumbernya bisa model yang tidak kita punya."
    print(f"{len(d)} baris unik | kelas mayoritas {base:.3f}\n{verdict}")
    return r
