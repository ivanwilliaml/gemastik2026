"""Bangun satu kernel CPU yang menjalankan semua notebook dengan USE_PRETRAINED=False.

Folder model dialihkan ke path yang tidak ada, jadi notebook yang diam-diam memuat model akan gagal.
Hasil: /kaggle/working/uji/hasil.csv dan notebook per task.

Pemakaian: python uji_tanpa_pretrained.py [--only 08 10]
lalu: kaggle kernels push -p %TEMP%\\gk\\uji -t 21600
"""
import argparse
import base64
import json
import re
import shutil
import zlib

from build import KERNEL_DIR, SRC_DIR, format_code, parse_percent, to_notebook
from tasks import OWNER, PACK_SLUG, TASKS

SLUG = "gemastik-nlp-uji-tanpa-pretrained"

RUNNER = """
import base64, json, pathlib, subprocess, sys, time, zlib
import pandas as pd

PAYLOADS = json.loads(zlib.decompress(base64.b64decode("{payload}")))
base = pathlib.Path("/kaggle/working/uji")
rows = []
for task_id, text in PAYLOADS.items():
    folder = base / task_id
    (folder / "hasil").mkdir(parents=True, exist_ok=True)
    source = folder / "_sumber.ipynb"
    source.write_text(text, encoding="utf-8")
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "papermill", str(source), str(folder / f"{{task_id}}.ipynb"),
         "--no-progress-bar", "--cwd", str(folder), "-k", "python3"],
        capture_output=True, text=True,
    )
    source.unlink()
    rows.append({{"id": task_id, "kode": result.returncode, "detik": round(time.time() - started),
                  "pesan": result.stderr[-600:] if result.returncode else ""}})
    pd.DataFrame(rows).to_csv(base / "hasil.csv", index=False)
    print(task_id, result.returncode, rows[-1]["detik"], flush=True)
"""


def patched_notebook(task: dict) -> str:
    cells = format_code(parse_percent(SRC_DIR / f"{task['id']}.py"))
    flags, models = 0, 0
    patched = []
    for kind, body in cells:
        body, n_flag = re.subn(r"^USE_PRETRAINED = True", "USE_PRETRAINED = False", body, flags=re.M)
        body, n_model = re.subn(r"^MODEL_DIR = .*$", 'MODEL_DIR = Path("/tmp/tidak_ada_model")', body, flags=re.M)
        flags, models = flags + n_flag, models + n_model
        patched.append((kind, body))
    already_off = any(re.search(r"^USE_PRETRAINED = False", body, flags=re.M) for _, body in patched)
    if not already_off or models != 1:
        raise SystemExit(f"{task['id']}: flag={flags} model_dir={models}")
    return json.dumps(to_notebook(patched), ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    tasks = [t for t in TASKS if not t["id"].startswith("00") and (not args.only or t["id"][:2] in args.only)]
    payload = {t["id"]: patched_notebook(t) for t in tasks}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")
    kernel = KERNEL_DIR / "uji"
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    (kernel / "uji.ipynb").write_text(json.dumps(to_notebook([("code", RUNNER.format(payload=encoded))])), "utf-8")
    meta = {
        "id": f"{OWNER}/{SLUG}",
        "title": "Gemastik NLP Uji Tanpa Pretrained",
        "code_file": "uji.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": sorted({d for t in tasks for d in t.get("datasets", [])}),
        "kernel_sources": [f"{OWNER}/{PACK_SLUG}"],
        "competition_sources": [],
        "model_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(f"ok {len(tasks)} notebook, {len(encoded) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
