"""Lingkungan offline, pengatur waktu, dan ringkasan run."""
import json
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

# Harus diset sebelum transformers diimpor, supaya tidak pernah mencoba internet.
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    TQDM_DISABLE="1",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
    PYTHONWARNINGS="ignore",
)
warnings.filterwarnings("ignore")
sys.modules.setdefault("tensorflow", None)  # tidak dipakai, mencegah log TensorFlow

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.max_colwidth", 110, "display.max_rows", 30, "display.width", 150)
plt.rcParams.update(
    {"figure.figsize": (8, 3.5), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
     "axes.spines.top": False, "axes.spines.right": False}
)

RESULT_DIR = Path("hasil")
RESERVE_SHARE = 0.1  # bagian waktu yang disisakan untuk post-processing dan menyimpan hasil


class Clock:
    """Mencatat waktu terpakai, dan menjawab apakah suatu pekerjaan masih muat."""

    def __init__(self, minutes: float, seed: int):
        self.budget = minutes * 60
        self.seed = seed
        self.started = time.time()
        self.stages: dict[str, float] = {}

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def fits(self, seconds: float) -> bool:
        return self.elapsed + seconds <= self.budget * (1 - RESERVE_SHARE)


CLOCK: Clock | None = None


class Notes:
    """Mengumpulkan temuan dan keputusan. Tiap sel EDA menambah sendiri, jadi boleh dicopot."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, finding: str, decision: str, severity: int = 0) -> None:
        """severity 0 = info, 1 = perlu diingat, 2 = masalah, 3 = masalah berat."""
        self.rows.append({"temuan": finding, "keputusan": decision, "bobot": severity})
        print(f"- {finding} -> {decision}")

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).sort_values("bobot", ascending=False, kind="stable").reset_index(drop=True)

    @property
    def biggest(self) -> str:
        worst = [r for r in self.rows if r["bobot"] >= 2]
        return max(worst, key=lambda r: r["bobot"])["temuan"] if worst else "Tidak ada masalah besar yang terdeteksi"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def start(minutes: float, seed: int = 42) -> Clock:
    """Mulai pengatur waktu. Dipanggil sekali, tepat setelah config."""
    global CLOCK
    CLOCK = Clock(minutes, seed)
    RESULT_DIR.mkdir(exist_ok=True)
    seed_everything(seed)
    print(f"device={device()} | batas waktu={minutes} menit | seed={seed}")
    return CLOCK


@contextmanager
def stage(name: str):
    """Catat durasi satu tahap."""
    begin = time.time()
    yield
    seconds = time.time() - begin
    if CLOCK is not None:
        CLOCK.stages[name] = round(seconds, 1)
    print(f"[waktu] {name}: {seconds:.0f} detik")


def finish(task: str, data, board, needs_gpu: bool = True) -> None:
    """Simpan tabel perbandingan dan cetak ringkasan run."""
    used_gpu = False
    try:
        import torch

        used_gpu = torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0
    except ImportError:
        pass
    total = int(CLOCK.elapsed)
    table = board.table()
    table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
    summary = {"task": task, "n_rows": data.n_rows, "total_detik": total, "gpu": used_gpu,
               "tahap_detik": CLOCK.stages, "metrik": table.to_dict("records")}
    (RESULT_DIR / "ringkasan.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    note = "" if needs_gpu else " gpu_tidak_wajib"
    print(f"RINGKASAN_RUN task={task} n_rows={data.n_rows} total_detik={total} gpu={used_gpu}{note}")
