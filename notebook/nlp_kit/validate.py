"""Metrik, fold, alat ukur noise, dan tabel perbandingan."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .text import adversarial_scores

METRICS = {
    "macro_f1": lambda y, p, w=None: f1_score(y, p, average="macro", zero_division=0, sample_weight=w),
    "weighted_f1": lambda y, p, w=None: f1_score(y, p, average="weighted", zero_division=0, sample_weight=w),
    "f1": lambda y, p, w=None: f1_score(y, p, average="binary", zero_division=0, sample_weight=w),
    "accuracy": lambda y, p, w=None: accuracy_score(y, p, sample_weight=w),
}


def get_metric(name: str):
    if name not in METRICS:
        raise KeyError(f"Metrik '{name}' belum ada. Pilihan: {list(METRICS)}")
    return METRICS[name]


def check_metrics() -> None:
    """Uji metrik pada contoh yang jawabannya sudah dihitung manual."""
    assert abs(METRICS["macro_f1"]([0, 0, 1, 1], [0, 1, 1, 1]) - (2 / 3 + 0.8) / 2) < 1e-9
    assert abs(METRICS["f1"]([0, 0, 1, 1], [0, 1, 1, 1]) - 0.8) < 1e-9
    assert METRICS["accuracy"]([0, 1], [0, 0]) == 0.5
    print("metrik lolos uji")


def make_folds(data, n_folds: int, seed: int) -> np.ndarray:
    """Stratified, dan grup yang sama tidak lintas fold. Jumlah fold turun kalau kelas terlalu kecil."""
    largest = int(data.labeled["label"].value_counts().max())
    groups = data.labeled["group"].nunique()
    n_splits = max(2, min(n_folds, largest, groups))
    if n_splits != n_folds:
        print(f"fold diturunkan dari {n_folds} ke {n_splits}")
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    folds = np.zeros(len(data.labeled), dtype=int)
    for fold, (_, val_idx) in enumerate(splitter.split(data.labeled, data.y, data.labeled["group"])):
        folds[val_idx] = fold
    print("baris per fold:", np.bincount(folds).tolist())
    return folds


def test_likeness_weights(train_texts: list[str], test_texts: list[str]) -> np.ndarray:
    """Baris latih yang mirip test mendapat bobot besar. Tidak memakai label test."""
    _, scores = adversarial_scores(train_texts, test_texts)
    head = scores[: len(train_texts)]
    return np.clip(head / np.clip(1 - head, 1e-3, None), 0.1, 10)


def prob_better(y, pred_a, pred_b, score, weights=None, seed: int = 42, n_boot: int = 500) -> float:
    """Peluang B lebih baik dari A, bootstrap berpasangan. Seri tidak dihitung menang."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(y), len(y))
        row_weight = None if weights is None else weights[pick]
        wins += score(y[pick], pred_b[pick], row_weight) > score(y[pick], pred_a[pick], row_weight)
    return wins / n_boot


@dataclass
class Preds:
    """Probabilitas satu model."""

    name: str
    oof: np.ndarray
    test: np.ndarray
    extra: dict = field(default_factory=dict)


class Leaderboard:
    """Mencatat skor semua model. Kolom test hanya muncul kalau label test tersedia."""

    def __init__(self, data, metric: str, seed: int = 42):
        self.data, self.metric_name, self.score, self.seed = data, metric, get_metric(metric), seed
        self.y = data.y
        self.weights = test_likeness_weights(data.labeled["clean"].tolist(), data.test["clean"].tolist())
        self.rows: list[dict] = []

    def interval(self, pred: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
        rng = np.random.default_rng(self.seed)
        scores = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(self.y), len(self.y))
            scores.append(self.score(self.y[pick], pred[pick]))
        return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

    def prob_better(self, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
        """Peluang B lebih baik dari A pada skor berbobot mirip test."""
        return prob_better(self.y, pred_a, pred_b, self.score, self.weights, self.seed)

    def add(self, preds: "Preds") -> "Preds":
        """Catat satu model dari probabilitasnya."""
        self.record(preds.name, preds.oof.argmax(1), preds.test.argmax(1))
        return preds

    def record(self, name: str, oof_pred: np.ndarray, test_pred: np.ndarray) -> None:
        low, high = self.interval(oof_pred)
        row = {"model": name, "oof": round(float(self.score(self.y, oof_pred)), 4), "oof_ci95": f"{low:.3f}-{high:.3f}",
               "oof_mirip_test": round(float(self.score(self.y, oof_pred, self.weights)), 4)}
        if self.data.has_test_labels:
            row["test"] = round(self._test_score(test_pred), 4)
        self.rows.append(row)
        print(row)

    def _test_score(self, test_pred: np.ndarray) -> float:
        known = self.data.y_test >= 0
        return float(self.score(self.data.y_test[known], test_pred[known]))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
