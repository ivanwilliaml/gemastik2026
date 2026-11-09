# title: Simpan OOF, test, id, config
# when: Setiap run. Satu perubahan per run, satu nama per run.
# avoid: Menimpa file run lama.
# tags: artefak oof log eksperimen
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

def save_run(name, oof=None, test=None, score=None, config=None, ids=None, folder="runs"):
    d = Path(folder) / name
    d.mkdir(parents=True, exist_ok=True)
    if oof is not None:
        np.save(d / "oof.npy", oof)
    if test is not None:
        np.save(d / "test.npy", test)
    if ids is not None:
        np.save(d / "ids.npy", np.asarray(ids))
    (d / "meta.json").write_text(json.dumps({"score": score, "config": config, "time": time.strftime("%H:%M:%S")}, default=str, indent=1))
    with open(Path(folder) / "log.csv", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')},{name},{score}\n")

def load_run(name, folder="runs"):
    d = Path(folder) / name
    return {p.stem: np.load(p, allow_pickle=True) for p in d.glob("*.npy")} | json.loads((d / "meta.json").read_text())

def leaderboard(folder="runs"):
    return pd.read_csv(Path(folder) / "log.csv", names=["waktu", "run", "skor"]).sort_values("skor", ascending=False)
