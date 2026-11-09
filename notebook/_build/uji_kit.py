"""Uji notebook versi nlp_kit di Kaggle pada beberapa bentuk data.

Path di dalam notebook sama seperti di lokal: kernel membuat pohon folder nlp/ yang sama, lalu
offline/ menunjuk ke paket model. Tidak ada path Kaggle di notebook maupun nlp_kit.

    A: SmSA, 500 berlabel + 4.500 kosong, test berlabel (notebook final)
    B: nama kolom beda, latih tsv, test xlsx tanpa label, ada id, waktu 6 menit
    C: biner, semua berlabel, csv latin-1, emoji berupa byte, metrik f1
    D: folder model tidak ada, kelas langka 2 baris, test tanpa label, 3 menit

Pemakaian: python uji_kit.py --shapes D --cpu
lalu: kaggle kernels push -p %TEMP%\\gk\\uji_kit -t 14400
"""
import argparse
import base64
import json
import shutil
import zlib
from pathlib import Path

from build import KERNEL_DIR, format_code, parse_percent, to_notebook
from tasks import OWNER, PACK_SLUG

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
SLUG = "gemastik-nlp-uji-kit"
NOTEBOOK = "08_semi_supervised"

RUNNER = r"""
import base64, json, os, pathlib, shutil, subprocess, sys, time, zlib
import pandas as pd

PACKED = json.loads(zlib.decompress(base64.b64decode("{payload}")))
SHAPES = {shapes}
root = pathlib.Path("/kaggle/working/nlp")
for rel, text in PACKED["files"].items():
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
pack = next(pathlib.Path("/kaggle/input").glob("**/pack_manifest.json")).parent
(root / "offline").symlink_to(pack)
smsa = pack / "data/indonlu/dataset/smsa_doc-sentiment-prosa"
read = lambda s: pd.read_csv(smsa / f"{{s}}_preprocess.tsv", sep="\t", header=None, names=["text", "label"])
pool = read("train").sample(frac=1, random_state=42).head(5000).reset_index(drop=True)
smsa_test = read("test")
nb_dir = root / "b_klasifikasi/08_semi_supervised"
nb_path = nb_dir / "_sumber.ipynb"
nb_dir.mkdir(parents=True, exist_ok=True)
nb_path.write_text(PACKED["notebook"], encoding="utf-8")


def hide_labels(frame, keep):
    picked = frame.groupby("label", group_keys=False).sample(n=None, frac=keep / len(frame), random_state=42)
    out = frame.copy()
    out["label_audit"] = out["label"]
    out.loc[~out.index.isin(picked.index), "label"] = None
    return out


def shape_a():
    folder = nb_dir / "data"
    folder.mkdir(exist_ok=True)
    hide_labels(pool, 500).to_csv(folder / "train.csv", index=False)
    smsa_test.to_csv(folder / "test.csv", index=False)
    return nb_dir, {{}}


def shape_b():
    folder = root / "uji/B"
    folder.mkdir(parents=True, exist_ok=True)
    train = hide_labels(pool.head(1500), 200).drop(columns="label_audit").rename(columns={{"text": "komentar", "label": "sentimen"}})
    train.to_csv(folder / "latih.tsv", sep="\t", index=False)
    test = smsa_test.drop(columns="label").rename(columns={{"text": "komentar"}})
    test.insert(0, "no", range(1000, 1000 + len(test)))
    test.to_excel(folder / "uji.xlsx", index=False)
    return folder, dict(TRAIN_PATH="latih.tsv", TEST_PATH="uji.xlsx", TEXT_COL="komentar", LABEL_COL="sentimen",
                        ID_COL="no", WAKTU_MENIT=6, MODELS=["indobert"])


def shape_c():
    folder = root / "uji/C"
    folder.mkdir(parents=True, exist_ok=True)
    source = next(p for p in pathlib.Path("/kaggle/input").glob("**/data.csv") if "abusive" in str(p).lower() or "hate" in str(p).lower())
    frame = pd.read_csv(source, encoding="latin-1").sample(frac=1, random_state=42)
    frame.head(2000)[["Tweet", "HS"]].to_csv(folder / "train.csv", index=False, encoding="latin-1")
    frame.iloc[2000:2500][["Tweet", "HS"]].to_csv(folder / "test.csv", index=False, encoding="latin-1")
    return folder, dict(TRAIN_PATH="train.csv", TEST_PATH="test.csv", TEXT_COL="Tweet", LABEL_COL="HS", METRIC="f1",
                        POSITIVE_CLASS="1", WAKTU_MENIT=10, MODELS=["indobertweet"])


def shape_d():
    folder = root / "uji/D"
    folder.mkdir(parents=True, exist_ok=True)
    train = hide_labels(pool.head(600), 300)
    rare = train[train["label"].notna()].index[:2]
    train.loc[rare, "label"] = "langka"
    train.drop(columns="label_audit").to_csv(folder / "train.csv", index=False)
    smsa_test.drop(columns="label").to_csv(folder / "test.csv", index=False)
    return folder, dict(TRAIN_PATH="train.csv", TEST_PATH="test.csv", MODEL_DIR="tidak_ada/models", WAKTU_MENIT=3)


def shape_s():
    # Uji cepat: data kecil, semua cabang transformer ikut jalan.
    folder = nb_dir / "data"
    folder.mkdir(exist_ok=True)
    hide_labels(pool.head(150), 60).to_csv(folder / "train.csv", index=False)
    smsa_test.head(30).to_csv(folder / "test.csv", index=False)
    return nb_dir, dict(WAKTU_MENIT=20)


def shape_t():
    # Uji cepat: semua berlabel, kolom beda, test tanpa label.
    folder = root / "uji/T"
    folder.mkdir(parents=True, exist_ok=True)
    pool.head(80).rename(columns={{"text": "komentar", "label": "sentimen"}}).to_csv(folder / "latih.tsv", sep="\t", index=False)
    test = smsa_test.head(30).drop(columns="label").rename(columns={{"text": "komentar"}})
    test.insert(0, "no", range(1000, 1030))
    test.to_excel(folder / "uji.xlsx", index=False)
    return folder, dict(TRAIN_PATH="latih.tsv", TEST_PATH="uji.xlsx", TEXT_COL="komentar", LABEL_COL="sentimen",
                        ID_COL="no", WAKTU_MENIT=20, MODELS=["indobert"])


rows = []
for shape in SHAPES:
    cwd, params = {{"A": shape_a, "B": shape_b, "C": shape_c, "D": shape_d, "S": shape_s, "T": shape_t}}[shape]()
    output = cwd / "08_semi_supervised.ipynb"
    begin = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "papermill", str(nb_path), str(output), "--no-progress-bar", "--cwd", str(cwd),
         "-k", "python3", "-y", json.dumps(params)], capture_output=True, text=True)
    rows.append({{"bentuk": shape, "kode": result.returncode, "detik": round(time.time() - begin), "pesan": result.stderr[-1500:] if result.returncode else ""}})
    pd.DataFrame(rows).to_csv(root / "uji_hasil.csv", index=False)
    print(rows[-1], flush=True)
(root / "offline").unlink()
nb_path.unlink()
shutil.rmtree(root / "nlp_kit" / "__pycache__", ignore_errors=True)
"""


def tagged_notebook() -> str:
    cells = format_code(parse_percent(BUILD / "src_v2" / f"{NOTEBOOK}.py"))
    notebook = to_notebook(cells)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("TRAIN_PATH"):
            cell["metadata"]["tags"] = ["parameters"]
    return json.dumps(notebook, indent=1, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["D", "B", "C", "A"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    files = {f"nlp_kit/{p.name}": p.read_text(encoding="utf-8") for p in (ROOT / "nlp_kit").glob("*.py")}
    payload = {"files": files, "notebook": tagged_notebook()}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode("utf-8"), 9)).decode("ascii")
    code = RUNNER.format(payload=encoded, shapes=json.dumps(args.shapes))
    kernel = KERNEL_DIR / "uji_kit"
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    (kernel / "uji_kit.ipynb").write_text(json.dumps(to_notebook([("code", code)])), "utf-8")
    meta = {
        "id": f"{OWNER}/{SLUG}",
        "title": "Gemastik NLP Uji Kit",
        "code_file": "uji_kit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not args.cpu,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text"],
        "kernel_sources": [f"{OWNER}/{PACK_SLUG}"],
        "competition_sources": [],
        "model_sources": [],
    }
    if not args.cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(f"ok, {len(encoded) / 1e3:.0f} KB, bentuk {args.shapes}")


if __name__ == "__main__":
    main()
