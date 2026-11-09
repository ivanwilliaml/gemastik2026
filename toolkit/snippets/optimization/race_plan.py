# title: Jam lomba dan pagar skala
# when: Setup. clock('tahap') mencetak menit berjalan dan gate berikutnya. scale_plan menurunkan eksperimen saat data besar.
# avoid: Menganggap ambang pagar berlaku di semua mesin. Ukur ulang di PC panitia; force=True bila RAM dan waktu cukup.
# tags: waktu gate deadline skala 500 ribu hashing folds sampel timer
import time
from datetime import datetime
import numpy as np

GATES = [(40, "G1 pilih cabang task"), (70, "G2 submission pertama"), (105, "G3 lanjut atau lewati BERT"), (150, "G4 proyeksi BERT"), (225, "G5 blend"), (240, "Freeze model"), (280, "Unggah final"), (300, "Lomba selesai")]
_T0 = [time.time()]

def start_clock(hhmm=None):
    """hhmm='13:00' saat lomba. Tanpa argumen, hitung dari sekarang."""
    if hhmm:
        h, m = map(int, hhmm.split(":"))
        _T0[0] = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()
    else:
        _T0[0] = time.time()

def clock(label=""):
    t = (time.time() - _T0[0]) / 60
    nxt = next((g for g in GATES if g[0] > t), None)
    print(f"[T+{t:.0f} menit] {label}" + (f" | berikutnya T{nxt[0]}: {nxt[1]}, sisa {nxt[0] - t:.0f} menit" if nxt else " | waktu habis"))
    return t

def scale_plan(n_rows, experiments, n_folds, medium=100_000, large=250_000, force=False):
    """Kembalikan (experiments, n_folds) yang aman untuk ukuran data. force=True mematikan pagar."""
    if force or n_rows <= medium:
        return experiments, n_folds
    if n_rows <= large:
        lighter = list(dict.fromkeys(("tfidf_word" if v in ("tfidf_union", "tfidf_char") else v, "sgd" if m in ("svc", "svc_cal", "rf", "mlp", "xgb", "lgbm") else m) for v, m in experiments))
        print(f"PAGAR SKALA {n_rows} baris: fitur char dibuang, {n_folds} fold jadi 3 -> {lighter}")
        return lighter, min(n_folds, 3)
    plan = [("hashing", "sgd"), ("tfidf_word", "sgd")]
    print(f"PAGAR SKALA {n_rows} baris: pakai {plan}, 3 fold. force=True bila yakin RAM dan waktu cukup.")
    return plan, min(n_folds, 3)

def sample_idx(n, k=100_000, seed=42):
    """Indeks acak untuk langkah mahal seperti bootstrap. Bila n <= k, semua baris."""
    return np.arange(n) if n <= k else np.random.default_rng(seed).choice(n, k, replace=False)
