"""Cetak output teks notebook hasil run, untuk ditinjau di terminal.

Pemakaian: python show.py 08 [--max 1500]
"""
import argparse
import json
import re
import sys
from pathlib import Path

from tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
NOISE = re.compile(r"Warning|it/s\]|s/it\]|Traceback", re.I)

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--max", type=int, default=1500)
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

task = next(t for t in TASKS if t["id"].startswith(args.task))
notebook = json.loads((ROOT / task["folder"] / f"{task['id']}.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    for out in cell.get("outputs", []):
        if out["output_type"] == "stream":
            text = "".join(out["text"])
            name = out["name"]
        elif out["output_type"] == "error":
            text = out["ename"] + ": " + out["evalue"]
            name = "ERROR"
        else:
            data = out.get("data", {})
            text = "".join(data.get("text/plain", []))
            name = "img" if "image/png" in data else "tampil"
        flag = " !!BERISIK" if NOISE.search(text) else ""
        print(f"--- sel {i} [{name}]{flag}\n{text[:args.max]}")
