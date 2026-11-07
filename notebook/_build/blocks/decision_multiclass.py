# %% [markdown]
# **Decision layer multi-kelas.** Pengali per kelas dicari untuk memaksimalkan macro-F1. Skor
# dilaporkan secara *nested*: pengali untuk tiap fold difit dari fold lain, jadi skornya jujur.

# %%
from sklearn.metrics import f1_score as _f1  # noqa: E402


def fit_class_multipliers(probs: np.ndarray, y: np.ndarray, rounds: int = 2) -> np.ndarray:
    """Coordinate ascent untuk pengali kelas yang memaksimalkan macro-F1."""
    multipliers = np.ones(probs.shape[1])
    grid = np.arange(0.4, 2.55, 0.1)
    for _ in range(rounds):
        for cls in range(probs.shape[1]):
            scores = []
            for value in grid:
                trial = multipliers.copy()
                trial[cls] = value
                scores.append(_f1(y, (probs * trial).argmax(1), average="macro", zero_division=0))
            multipliers[cls] = grid[int(np.argmax(scores))]
    return multipliers


def nested_multiplier_predictions(probs: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Prediksi OOF: tiap fold memakai pengali yang difit dari fold lain."""
    preds = np.zeros(len(y), dtype=int)
    for fold in np.unique(folds):
        fit_mask = folds != fold
        multipliers = fit_class_multipliers(probs[fit_mask], y[fit_mask])
        preds[~fit_mask] = (probs[~fit_mask] * multipliers).argmax(1)
    return preds
