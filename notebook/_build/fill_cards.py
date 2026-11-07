"""Tulis isi cards.py ke sumber notebook dan notebook hasil run, tanpa menjalankan ulang.

Urutan placeholder ISI_SETELAH_EDA di tiap notebook: kartu masalah, temuan EDA, poin presentasi.
Pemakaian: python fill_cards.py
"""
import json
from pathlib import Path

from cards import CARDS
from tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent / "src"
MARK = "ISI_SETELAH_EDA"


def sections(info: dict) -> list[str]:
    kartu = ("| Bagian | Isi |\n|---|---|\n"
             f"| **Task** | {info['task']} |\n| **Unit prediksi** | {info['unit']} |\n"
             f"| **Metrik** | {info['metric']} |\n| **Masalah terbesar** | {info['problem']} |\n"
             f"| **Data contoh** | {info['data']} |")
    temuan = "| Temuan | Keputusan |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in info["findings"])
    poin = "\n".join(f"- {p}" for p in info["points"])
    return [kartu, temuan, poin]


def fill_source(path: Path, parts: list[str]) -> int:
    text = path.read_text(encoding="utf-8")
    count = 0
    for part in parts:
        commented = "\n".join(f"# {line}" for line in part.split("\n"))
        if f"# {MARK}" in text:
            text = text.replace(f"# {MARK}", commented, 1)
            count += 1
    path.write_text(text, encoding="utf-8")
    return count


def fill_notebook(path: Path, parts: list[str]) -> int:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    remaining = iter(parts)
    count = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and MARK in source:
            cell["source"] = source.replace(MARK, next(remaining)).splitlines(keepends=True)
            count += 1
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    return count


missing = []
for task in TASKS:
    number = task["id"][:2]
    if number == "00":
        continue
    if number not in CARDS:
        missing.append(number)
        continue
    parts = sections(CARDS[number])
    in_source = fill_source(SRC / f"{task['id']}.py", parts)
    in_notebook = fill_notebook(ROOT / task["folder"] / f"{task['id']}.ipynb", parts)
    print(f"{task['id']}: sumber {in_source}, notebook {in_notebook}")
print("belum ada kartu:", missing)
