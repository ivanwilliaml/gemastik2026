"""Memuat data apa pun ke satu bentuk yang sama: TaskData."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_COL = "label_audit"  # opsional, label asli baris tanpa label, hanya untuk audit


def load_table(path) -> pd.DataFrame:
    """Baca csv, tsv, xlsx, json, jsonl, atau parquet. Encoding dan pemisah dicari otomatis."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ada: {path.resolve()}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if suffix == ".tsv":
                return pd.read_csv(path, sep="\t", encoding=encoding)
            frame = pd.read_csv(path, encoding=encoding)
            if frame.shape[1] == 1:  # bisa jadi pemisahnya bukan koma, dilihat dari header
                separator = next((s for s in ";\t|" if s in str(frame.columns[0])), None)
                if separator:
                    frame = pd.read_csv(path, sep=separator, encoding=encoding)
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Encoding tidak dikenali: {path}")


def _require(frame: pd.DataFrame, column, path) -> None:
    if column not in frame.columns:
        raise KeyError(f"Kolom '{column}' tidak ada di {path}. Kolom yang tersedia: {list(frame.columns)}")


def _join_text(frame: pd.DataFrame, columns) -> pd.Series:
    """Satu kolom teks, atau beberapa kolom yang digabung, misalnya judul dan isi."""
    if isinstance(columns, str):
        return frame[columns].fillna("").astype(str).str.strip()
    return frame[list(columns)].fillna("").astype(str).agg(" . ".join, axis=1).str.strip()


def _label_to_text(value) -> str:
    """Samakan bentuk label. 1.0 dari kolom yang ada kosongnya menjadi '1'."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _sorted_classes(values) -> list[str]:
    unique = sorted(set(values))
    try:
        return sorted(unique, key=float)
    except ValueError:
        return unique


@dataclass
class TaskData:
    """Bentuk data yang dipakai semua fungsi lain."""

    labeled: pd.DataFrame  # text, label, group opsional
    unlabeled: pd.DataFrame  # text, audit opsional
    test: pd.DataFrame  # id, text, label opsional
    classes: list[str]
    label_name: str
    id_name: str

    @property
    def y(self) -> np.ndarray:
        return self.labeled["label"].map(self.class_id).to_numpy()

    @property
    def class_id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}

    @property
    def has_test_labels(self) -> bool:
        return "label" in self.test.columns

    @property
    def y_test(self) -> np.ndarray | None:
        if not self.has_test_labels:
            return None
        return self.test["label"].map(self.class_id).fillna(-1).astype(int).to_numpy()

    @property
    def n_rows(self) -> int:
        return len(self.labeled) + len(self.unlabeled)


def load_task(train_path, test_path, text_col, label_col, id_col=None, group_col=None,
              positive_class=None) -> TaskData:
    """Muat data klasifikasi. Label kosong otomatis menjadi data tanpa label."""
    train, test = load_table(train_path), load_table(test_path)
    for column in [text_col] if isinstance(text_col, str) else text_col:
        _require(train, column, train_path)
        _require(test, column, test_path)
    _require(train, label_col, train_path)
    if id_col:
        _require(test, id_col, test_path)
    if group_col:
        _require(train, group_col, train_path)

    frame = pd.DataFrame({"text": _join_text(train, text_col)})
    filled = train[label_col].notna() & (train[label_col].astype(str).str.strip() != "")
    frame["label"] = [_label_to_text(v) if ok else None for v, ok in zip(train[label_col], filled)]
    if group_col:
        frame["group"] = train[group_col].astype(str).to_numpy()
    if AUDIT_COL in train.columns:
        frame["audit"] = train[AUDIT_COL].map(lambda v: None if pd.isna(v) else _label_to_text(v))

    keep = frame["text"].str.len() > 0
    labeled = frame[filled & keep].reset_index(drop=True)
    unlabeled = frame[~filled & keep].drop(columns=["label"]).reset_index(drop=True)
    if labeled.empty:
        raise ValueError("Tidak ada baris berlabel. Untuk data tanpa label, pakai folder A_tanpa_label.")

    # Baris test tidak boleh dibuang, karena jumlah baris submission harus utuh.
    test_frame = pd.DataFrame(
        {"id": test[id_col].to_numpy() if id_col else np.arange(len(test)), "text": _join_text(test, text_col)}
    )
    if label_col in test.columns and test[label_col].notna().any():
        test_frame["label"] = [_label_to_text(v) for v in test[label_col]]

    classes = _sorted_classes(labeled["label"])
    if positive_class is not None and len(classes) == 2:
        positive = str(positive_class)
        if positive not in classes:
            raise ValueError(f"POSITIVE_CLASS '{positive}' tidak ada di kelas {classes}")
        classes = [c for c in classes if c != positive] + [positive]
    data = TaskData(labeled, unlabeled, test_frame, classes, label_col, id_col or "id")
    print(f"berlabel={len(labeled)} | tanpa_label={len(unlabeled)} | test={len(test_frame)} | "
          f"kelas={classes} | label_test={'ada' if data.has_test_labels else 'tidak ada'}")
    return data
