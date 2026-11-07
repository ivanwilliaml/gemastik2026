# title: Bootstrap noise floor dan banding berpasangan
# when: T70-T90. Tentukan batas terima sebelum mengubah apa pun. Grup: cluster bootstrap.
# avoid: Mengejar gerakan kecil leaderboard.
# tags: bootstrap se noise paired kurva_belajar
import numpy as np

def _sample(n, groups, rng):
    if groups is None:
        return rng.integers(0, n, n)
    g = np.asarray(groups)
    u = np.unique(g)
    pick = rng.choice(u, len(u))
    pos = {k: np.where(g == k)[0] for k in u}
    return np.concatenate([pos[k] for k in pick])

def noise_floor(y, pred, score_fn, n_boot=300, groups=None, seed=42):
    """Kembalikan (skor, SE, batas 2,5%, batas 97,5%). groups untuk cluster bootstrap."""
    y, pred, rng = np.asarray(y), np.asarray(pred), np.random.default_rng(seed)
    s = [score_fn(y[i], pred[i]) for i in (_sample(len(y), groups, rng) for _ in range(n_boot))]
    return round(score_fn(y, pred), 4), round(float(np.std(s)), 4), round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)

def paired_gain(y, pred_a, pred_b, score_fn, n_boot=300, groups=None, seed=42):
    """B dibanding A pada baris yang sama. Terima B bila gain > 2*se."""
    y, a, b, rng = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b), np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = _sample(len(y), groups, rng)
        d.append(score_fn(y[i], b[i]) - score_fn(y[i], a[i]))
    return dict(gain=round(score_fn(y, b) - score_fn(y, a), 4), se=round(float(np.std(d)), 4), p_b_lebih_baik=round(float(np.mean(np.array(d) > 0)), 3))

def learning_curve_oof(fit_score, n_rows, shares=(.1, .25, .5, 1.0), seed=42):
    """fit_score(idx) melatih pada idx dan mengembalikan skor validasi tetap."""
    order = np.random.default_rng(seed).permutation(n_rows)
    return {s: fit_score(order[:int(n_rows * s)]) for s in shares}
