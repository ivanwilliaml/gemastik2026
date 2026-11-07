# %% [markdown]
# **Metrik, interval noise, dan pencatat hasil.** Metrik diuji dulu pada contoh yang jawabannya
# dihitung manual. `record` mencatat skor OOF dan skor holdout untuk tiap pendekatan.

# %%
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


assert abs(macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) - (2 / 3 + 0.8) / 2) < 1e-9


def bootstrap_interval(y_true, y_pred, score_fn=macro_f1, n_boot: int = 300) -> str:
    """Interval 95%. Selisih di dalam interval ini belum tentu perbaikan nyata."""
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(y_true), len(y_true))
        scores.append(score_fn(y_true[pick], y_pred[pick]))
    return f"{np.percentile(scores, 2.5):.3f}-{np.percentile(scores, 97.5):.3f}"


def make_folds(target, groups, n_folds: int) -> np.ndarray:
    """StratifiedGroupKFold: duplikat tidak lintas fold, proporsi kelas terjaga."""
    splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED)
    folds = np.zeros(len(target), dtype=int)
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(target)), target, groups)):
        folds[val_idx] = fold
    return folds


results: list[dict] = []


def record(name: str, oof_pred, holdout_pred, score_fn=macro_f1) -> None:
    """Catat skor OOF dan holdout. Holdout hanya untuk laporan, bukan untuk memilih model."""
    results.append(
        {"model": name, "oof": round(score_fn(y, oof_pred), 4),
         "oof_ci95": bootstrap_interval(y, oof_pred, score_fn),
         "holdout": round(score_fn(y_holdout, holdout_pred), 4)}
    )
    print(results[-1])
