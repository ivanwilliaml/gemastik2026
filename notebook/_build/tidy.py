"""Rapikan notebook hasil papermill: buang metadata eksekusi supaya file bersih.

Pemakaian: python tidy.py jalur/notebook.ipynb
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
notebook = json.loads(path.read_text(encoding="utf-8"))
notebook["metadata"].pop("papermill", None)
for cell in notebook["cells"]:
    cell["metadata"] = {}
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
