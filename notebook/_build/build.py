"""Generator notebook cheat sheet.

Sumber tiap notebook ada di src/<id>.py dalam format percent:
    # %% [markdown]   -> sel markdown
    # %%              -> sel kode
    # %% include nama -> sel-sel dari blocks/nama.py disisipkan di sini

Pemakaian:
    python build.py --mode cepat             # bangun semua notebook yang sumbernya ada
    python build.py --mode penuh --only 08 01
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tasks import OWNER, PACK_SLUG, TASKS

BUILD_DIR = Path(__file__).resolve().parent
ROOT = BUILD_DIR.parent
SRC_DIR = BUILD_DIR / "src"
BLOCK_DIR = BUILD_DIR / "blocks"
KERNEL_DIR = Path(tempfile.gettempdir()) / "gk"  # path pendek, batas 260 karakter di Windows

CELL_MARK = re.compile(r"^# %%(?P<rest>.*)$")


def parse_percent(path: Path) -> list[tuple[str, str]]:
    """Pecah file percent menjadi daftar (jenis, isi); include diselesaikan rekursif."""
    cells: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        mark = CELL_MARK.match(line)
        if not mark:
            if current is not None:
                current[1].append(line)
            continue
        rest = mark.group("rest").strip()
        if rest.startswith("include "):
            current = None
            cells.extend(
                (kind, body.split("\n"))
                for kind, body in parse_percent(BLOCK_DIR / f"{rest.split()[1]}.py")
            )
            continue
        current = ("markdown" if rest == "[markdown]" else "code", [])
        cells.append(current)
    return [(kind, _clean(kind, lines)) for kind, lines in cells]


def _clean(kind: str, lines: list[str]) -> str:
    text = "\n".join(lines).strip("\n")
    if kind == "markdown":
        text = "\n".join(re.sub(r"^# ?", "", ln) for ln in text.split("\n"))
    return text


def format_code(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Format sel kode dengan ruff dan cek pyflakes pada gabungan kodenya."""
    formatted = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (kind, body) in enumerate(cells):
            if kind == "markdown":
                formatted.append((kind, body))
                continue
            cell_file = Path(tmp) / f"cell_{i}.py"
            cell_file.write_text(body + "\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "format", "--line-length", "100", str(cell_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise SystemExit(f"Gagal format sel {i}:\n{body}\n{result.stderr}")
            formatted.append((kind, cell_file.read_text(encoding="utf-8").strip("\n")))
        joined = Path(tmp) / "joined.py"
        joined.write_text(
            "\n\n".join(b for k, b in formatted if k == "code") + "\n", encoding="utf-8"
        )
        lint = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F", "--output-format", "concise",
             "--no-cache", str(joined)],
            capture_output=True,
            text=True,
        )
        if lint.returncode != 0:
            raise SystemExit("Lint gagal:\n" + lint.stdout)
    return formatted


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for i, (kind, body) in enumerate(cells):
        cell = {
            "cell_type": kind,
            "id": f"cell-{i:03d}",
            "metadata": {},
            "source": body.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def runner_notebook(task_id: str, notebook_text: str) -> dict:
    """Notebook pembungkus: menjalankan notebook asli dengan papermill.

    CLI Kaggle tidak bisa mengunduh notebook beserta output-nya. Dengan papermill, hasil eksekusi
    tersimpan sebagai file di /kaggle/working, jadi bisa diunduh.
    """
    payload = base64.b64encode(notebook_text.encode("utf-8")).decode("ascii")
    code = (
        "import base64, pathlib, subprocess, sys\n"
        f'payload = "{payload}"\n'
        'source = pathlib.Path("/kaggle/working/_sumber.ipynb")\n'
        "source.write_bytes(base64.b64decode(payload))\n"
        "result = subprocess.run([sys.executable, '-m', 'papermill', str(source), "
        f"'/kaggle/working/{task_id}.ipynb', '--no-progress-bar', '--cwd', '/kaggle/working', "
        "'-k', 'python3'])\n"
        "source.unlink()\n"
        "if result.returncode:\n"
        "    raise RuntimeError('Notebook gagal, lihat output notebook')\n"
    )
    return to_notebook([("code", code)])


def build_task(task: dict, mode: str) -> None:
    src = SRC_DIR / f"{task['id']}.py"
    if not src.exists():
        return
    cells = format_code(parse_percent(src))
    cells = [
        (k, re.sub(r'^MODE = "\w+"', f'MODE = "{mode}"', b, flags=re.M)) for k, b in cells
    ]
    notebook = to_notebook(cells)
    notebook_text = json.dumps(notebook, indent=1, ensure_ascii=False)

    kernel = KERNEL_DIR / task["id"]
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    code_file = f"{task['id']}.ipynb"
    is_pack = task["id"].startswith("00")
    pushed = notebook_text if is_pack else json.dumps(runner_notebook(task["id"], notebook_text))
    (kernel / code_file).write_text(pushed, "utf-8")
    meta = {
        "id": f"{OWNER}/{task['slug']}",
        "title": task["kernel_title"],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": task.get("gpu", True),
        "enable_tpu": False,
        "enable_internet": task.get("internet", False),
        "dataset_sources": task.get("datasets", []),
        "kernel_sources": [] if task["id"].startswith("00") else [f"{OWNER}/{PACK_SLUG}"],
        "competition_sources": [],
        "model_sources": [],
    }
    if task.get("gpu", True):
        meta["machine_shape"] = "NvidiaTeslaT4"
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")

    target = ROOT / task["folder"]
    target.mkdir(parents=True, exist_ok=True)
    target_nb = target / code_file
    if not target_nb.exists():
        target_nb.write_text(notebook_text, "utf-8")
    print(f"ok  {task['id']}  ({len(cells)} sel)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cepat", choices=["cepat", "penuh"])
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    for task in TASKS:
        if args.only and task["id"][:2] not in args.only:
            continue
        build_task(task, args.mode)


if __name__ == "__main__":
    main()
