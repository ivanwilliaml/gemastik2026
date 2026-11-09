# title: Ukur throughput dan proyeksi waktu
# when: T90-T105 sebelum komit. Kill rule pada 25% anggaran.
# avoid: Estimasi lama GTX 1650.
# tags: throughput waktu proyeksi kill_rule timer
import time
from contextlib import contextmanager

def measure(step_fn, n_batches=50, batch=32, skip=5):
    """step_fn() menjalankan satu batch. Kembalikan sampel per detik."""
    for _ in range(skip):
        step_fn()
    t = time.time()
    for _ in range(n_batches):
        step_fn()
    return batch * n_batches / max(time.time() - t, 1e-9)

def project(n_rows, rate, epochs=1, safety=2.0):
    m = n_rows * epochs / rate * safety / 60
    print(f"proyeksi {m:.0f} menit untuk {n_rows} baris x {epochs} epoch (cadangan x{safety})")
    return m

def kill_bert(score_bert, score_tfidf, elapsed_share, prev_score=None):
    """Pada 25% anggaran: di bawah TF-IDF dan tidak naik berarti kecilkan atau hentikan."""
    return elapsed_share >= .25 and score_bert < score_tfidf and (prev_score is None or score_bert <= prev_score)

@contextmanager
def stage(name, log={}):
    t = time.time()
    yield
    log[name] = round(time.time() - t, 1)
    print(f"[waktu] {name}: {log[name]} detik")
