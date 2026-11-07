"""Rekap semua notebook: error, runtime, dan tabel perbandingan.

Pemakaian: python recap.py [--ids 24 25]
"""
import argparse
import json
import sys
from pathlib import Path

from tasks import TASKS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs="*", default=[])
args = parser.parse_args()

for task in TASKS:
    if task["id"].startswith("00") or (args.ids and task["id"][:2] not in args.ids):
        continue
    folder = ROOT / task["folder"]
    print(f"\n=== {task['id']}")
    notebook = json.loads((folder / f"{task['id']}.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        for out in cell.get("outputs", []):
            if out["output_type"] == "error":
                print("ERROR", out["ename"], out["evalue"][:400])
    summary = folder / "hasil" / "ringkasan.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        print(f"mode={data['mode']} baris={data['n_rows']} detik={data['total_detik']} gpu={data['gpu']}")
    table = folder / "hasil" / "perbandingan.csv"
    if table.exists():
        print(table.read_text(encoding="utf-8").strip()[:1100])
