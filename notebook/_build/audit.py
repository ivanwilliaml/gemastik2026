"""Audit notebook hasil run Kaggle. Hanya membaca file, tidak menjalankan model.

Pemakaian: python audit.py            # semua notebook di folder cheat sheet
           python audit.py --only 08 01
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = [
    "Kartu Masalah", "Kapan Pakai", "Config", "EDA", "Preprocessing", "Validasi",
    "Baseline", "Model Utama", "Post-processing", "Post-analysis", "Keluaran",
]
MAX_OUTPUT_LINES = 60
MAX_RUNTIME_SECONDS = 2 * 3600
NOISE = re.compile(r"Warning|it/s\]|s/it\]|Traceback", re.I)


def output_text(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", [])))
        else:
            parts.append("".join(out.get("data", {}).get("text/plain", [])))
    return "\n".join(parts)


def audit_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    problems = []

    if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
        problems.append("ada sel error")
    if not any(c.get("outputs") for c in code):
        problems.append("tidak ada output, belum dijalankan")

    headings = ["".join(c["source"]) for c in cells if c["cell_type"] == "markdown"]
    positions = []
    for section in SECTIONS:
        hit = [i for i, h in enumerate(headings) if re.search(rf"^## .*{section}", h, re.M)]
        if not hit:
            problems.append(f"bagian hilang: {section}")
        else:
            positions.append(hit[0])
    if positions != sorted(positions):
        problems.append("urutan bagian salah")

    text = "\n".join(headings)
    if "ISI_SETELAH_EDA" in text:
        problems.append("kartu masalah belum diisi")

    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and i and cells[i - 1]["cell_type"] != "markdown":
            problems.append(f"sel kode {i} tanpa penjelasan")
            break

    for cell in code:
        out = output_text(cell)
        if out.count("\n") > MAX_OUTPUT_LINES:
            problems.append("output kebanyakan")
            break
    if any(NOISE.search(output_text(c)) for c in code):
        problems.append("output berisik (warning atau progress bar)")

    joined = "\n".join(output_text(c) for c in code)
    if not re.search(r"RINGKASAN_RUN", joined):
        problems.append("ringkasan run tidak tercetak")
    else:
        match = re.search(r"total_detik=(\d+)", joined)
        if match and int(match.group(1)) > MAX_RUNTIME_SECONDS:
            problems.append(f"runtime {match.group(1)} detik melebihi batas")
        if re.search(r"gpu=False", joined) and "gpu_tidak_wajib" not in joined:
            problems.append("GPU tidak terpakai")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    failed = 0
    for task in TASKS:
        if task["id"].startswith("00"):
            continue
        if args.only and task["id"][:2] not in args.only:
            continue
        path = ROOT / task["folder"] / f"{task['id']}.ipynb"
        if not path.exists():
            print(f"BELUM ADA  {task['id']}")
            failed += 1
            continue
        problems = audit_notebook(path)
        failed += bool(problems)
        print(("OK         " if not problems else "MASALAH    ") + task["id"], *problems, sep=" | ")
    print(f"\nnotebook bermasalah: {failed}")


if __name__ == "__main__":
    main()
