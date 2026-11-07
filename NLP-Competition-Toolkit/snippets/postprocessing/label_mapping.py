# title: Encode, decode, dan validator submission
# when: Setiap submit: kolom, jumlah baris, urutan id, NaN, nilai label, SHA-256.
# avoid: Mengedit di Excel.
# tags: submission label_encoder validator sha256 decode
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def encode_labels(y):
    """Kembalikan (y_int, kelas). Decode: kelas[pred]."""
    classes, yi = np.unique(np.asarray(y).astype(str), return_inverse=True)
    return yi, classes

def write_submission(sample_path, preds, out="submission.csv", pred_col=None, id_col=None, allowed=None):
    """Salin struktur contoh submission, isi prediksi, validasi, dan cetak hash."""
    s = pd.read_csv(sample_path)
    id_col = id_col or s.columns[0]
    pred_col = pred_col or s.columns[-1]
    assert len(s) == len(preds), f"jumlah baris {len(preds)} != contoh {len(s)}"
    s[pred_col] = preds
    assert s[pred_col].notna().all(), "ada NaN"
    assert s[id_col].is_unique, "id ganda"
    if allowed is not None:
        bad = set(s[pred_col]) - set(allowed)
        assert not bad, f"nilai asing: {bad}"
    s.to_csv(out, index=False)
    print(out, s.shape, "sha256", hashlib.sha256(Path(out).read_bytes()).hexdigest()[:16], s[pred_col].value_counts(normalize=True).head(5).round(3).to_dict())
    return s
