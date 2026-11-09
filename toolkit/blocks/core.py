"""Kontrak bersama. Tidak memasang paket atau menjalankan model saat disalin."""
import json
import random
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_RUNNERS = {}
ENCODERS = {}


def encode_e5(texts, cfg, role='query'):
    assert 'e5' in ENCODERS, 'Pasang komponen encoder E5 melalui perakit.'
    return ENCODERS['e5'](texts,cfg,role)


def register(name):
    def decorate(fn):
        MODEL_RUNNERS[name] = fn
        return fn
    return decorate


def read_table(path, encoding="utf-8", string_cols=()):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Isi CFG['data_file'] dengan file Kaggle yang ada: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", encoding=encoding,
        dtype={col:str for col in string_cols} or None)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def target_array(frame, cfg):
    return frame[cfg['label_cols']].to_numpy(float) if cfg['task']=='multilabel' else frame.y.to_numpy()


def require_kaggle(gpu=False):
    assert Path('/kaggle/input').exists(), "Jalankan eksperimen di Kaggle."
    if gpu:
        import torch
        assert torch.cuda.is_available(), "Aktifkan GPU Kaggle."


def show_table(name, table):
    from IPython.display import display
    print("\n" + name)
    display(table.to_frame() if isinstance(table, pd.Series) else table)


def save_tables(tables, out, prefix=""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{prefix}{name}.csv", index=True)
