"""Cetak output ringkas per bagian notebook, sebagai bahan menulis kartu masalah.

Pemakaian: python ringkas.py 08 01 [--max 700]
"""
import argparse
import json
import re
import sys
from pathlib import Path

from tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
SKIP_SECTIONS = {"Kartu", "Kapan", "Config", "Keluaran"}
parser = argparse.ArgumentParser()
parser.add_argument("ids", nargs="+")
parser.add_argument("--max", type=int, default=700)
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for task in TASKS:
    if task["id"][:2] not in args.ids:
        continue
    print(f"\n===== {task['id']}")
    notebook = json.loads((ROOT / task["folder"] / f"{task['id']}.ipynb").read_text(encoding="utf-8"))
    section = ""
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = re.match(r"## \d+\. (\w+)", source)
            if heading:
                section = heading.group(1)
                print(f"-- {section}")
            continue
        if not section or section in SKIP_SECTIONS:
            continue
        for out in cell.get("outputs", []):
            if out["output_type"] == "stream":
                text = "".join(out["text"])
            elif "text/plain" in out.get("data", {}) and "image/png" not in out["data"]:
                text = "".join(out["data"]["text/plain"])
            else:
                continue
            text = re.sub(r"[ \t]{2,}", "  ", text)
            print(text[: args.max].rstrip())
