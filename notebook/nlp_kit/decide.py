"""Alat post-processing dan penyimpanan hasil. Keputusannya sendiri ada di notebook."""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix


@dataclass
class Final:
    """Keputusan akhir."""

    name: str
    oof_pred: np.ndarray
    test_pred: np.ndarray
    oof_probs: np.ndarray
    test_probs: np.ndarray


def fit_multipliers(probs, y, score, rounds: int = 2) -> np.ndarray:
    """Pengali per kelas dengan coordinate ascent."""
    multipliers = np.ones(probs.shape[1])
    grid = np.arange(0.4, 2.55, 0.1)
    for _ in range(rounds):
        for cls in range(probs.shape[1]):
            scores = []
            for value in grid:
                trial = multipliers.copy()
                trial[cls] = value
                scores.append(score(y, (probs * trial).argmax(1)))
            multipliers[cls] = grid[int(np.argmax(scores))]
    return multipliers


def nested_multiplier_pred(probs, y, folds, score) -> np.ndarray:
    """Tiap fold memakai pengali yang difit dari fold lain, jadi skornya jujur."""
    preds = np.zeros(len(y), dtype=int)
    for fold in np.unique(folds):
        fit = folds != fold
        preds[~fit] = (probs[~fit] * fit_multipliers(probs[fit], y[fit], score)).argmax(1)
    return preds


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Satu angka T yang meminimalkan log loss. T > 1 artinya model terlalu yakin."""
    logits = np.log(np.clip(probs, 1e-7, 1))

    def loss(temperature: float) -> float:
        scaled = logits / temperature
        scaled -= scaled.max(1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(1, keepdims=True))
        return -log_probs[np.arange(len(y)), y].mean()

    return float(minimize_scalar(loss, bounds=(0.2, 10), method="bounded").x)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-7, 1)) / temperature
    logits -= logits.max(1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(1, keepdims=True)


def calibration_table(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Keyakinan melawan akurasi per rentang, dan ECE."""
    confidence, correct = probs.max(1), probs.argmax(1) == y
    edges = pd.cut(confidence, np.linspace(0, 1, bins + 1), include_lowest=True)
    table = pd.DataFrame({"yakin": confidence, "benar": correct}).groupby(edges, observed=True).agg(
        baris=("benar", "size"), yakin=("yakin", "mean"), akurasi=("benar", "mean"))
    ece = float((table["baris"] / len(y) * (table["yakin"] - table["akurasi"]).abs()).sum())
    return table.round(3), ece


def em_prior(probs: np.ndarray, train_prior: np.ndarray, rounds: int = 100) -> np.ndarray:
    """Perkiraan proporsi kelas test tanpa label, metode Saerens."""
    train_prior = np.clip(train_prior, 1e-6, 1)
    prior = train_prior.copy()
    for _ in range(rounds):
        adjusted = probs * prior / train_prior
        adjusted /= np.clip(adjusted.sum(1, keepdims=True), 1e-12, None)
        updated = adjusted.mean(0)
        if np.abs(updated - prior).max() < 1e-5:
            return updated
        prior = updated
    return prior


def plot_confusion(classes, y, pred, title: str = "Confusion matrix OOF") -> None:
    labels = list(range(len(classes)))
    matrix = confusion_matrix(y, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=labels, yticks=labels, xticklabels=classes, yticklabels=classes,
           title=title, xlabel="prediksi", ylabel="asli")
    ax.grid(False)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image)
    plt.tight_layout()
    plt.show()


def save_outputs(data, final: Final, result_dir="hasil") -> pd.DataFrame:
    """Tulis submission dan OOF. Jumlah baris diperiksa supaya sesuai test."""
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    submission = pd.DataFrame({data.id_name: data.test["id"].to_numpy(),
                               data.label_name: [data.classes[i] for i in final.test_pred]})
    assert len(submission) == len(data.test) and submission[data.label_name].notna().all()
    submission.to_csv(result_dir / "submission.csv", index=False)
    oof = pd.DataFrame(final.oof_probs, columns=data.classes).assign(label=data.labeled["label"].to_numpy())
    oof.to_csv(result_dir / "oof.csv", index=False)
    print(f"tersimpan: {result_dir / 'submission.csv'} ({len(submission)} baris)")
    return submission.head(3)
