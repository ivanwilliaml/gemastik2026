# %% [markdown]
# # 00 Setup Offline
#
# Notebook ini **butuh internet** dan cukup dijalankan sekali sebelum lomba. Hasilnya satu paket
# berisi model, data tambahan, dan wheel library. Semua notebook task memakai paket ini sebagai
# input dan berjalan tanpa internet.
#
# **Isi notebook**
# 1. Config
# 2. Unduh model
# 3. Unduh data tambahan
# 4. Unduh wheel library
# 5. Laporan skema dataset
# 6. Manifest paket

# %% [markdown]
# ## 1. Config

# %%
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

OUT_DIR = Path("/kaggle/working")
MODEL_DIR = OUT_DIR / "models"
DATA_DIR = OUT_DIR / "data"
WHEEL_DIR = OUT_DIR / "wheels"
INPUT_DIR = Path("/kaggle/input")

MODELS = {
    "e5": "intfloat/multilingual-e5-base",
    "indobertweet": "indolem/indobertweet-base-uncased",
    "indobert": "indobenchmark/indobert-base-p1",
    "mdeberta_nli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "mt5_sum": "csebuetnlp/mT5_multilingual_XLSum",
    "marian_id_en": "Helsinki-NLP/opus-mt-id-en",
    "marian_en_id": "Helsinki-NLP/opus-mt-en-id",
    "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "llm": "Qwen/Qwen2.5-1.5B-Instruct",
}
GIT_REPOS = {
    "indonlu": "https://github.com/IndoNLP/indonlu.git",
    "stif": "https://github.com/haryoa/stif-indonesia.git",
    "kamus_alay": "https://github.com/nasalsabila/kamus-alay.git",
}
HF_DATASETS = {
    "alt": ("mutiyama/alt", "alt-parallel"),
    "stsb_id": ("LazarusNLP/stsb_mt_id", None),
    "indonli": ("afaji/indonli", None),
    "paracotta": ("Wikidepia/IndoParaCotta", None),
}
PACKAGES = [
    "umap-learn", "hdbscan", "rouge-score", "sacrebleu", "yake", "rank-bm25", "ruptures",
    "sentencepiece", "protobuf", "gensim", "jiwer", "sentence-transformers", "lightgbm",
]
KEEP_PATTERNS = [
    "*.json", "*.txt", "*.model", "*.spm", "*.safetensors", "pytorch_model*.bin",
    "1_Pooling/*",
]
SKIP_PATTERNS = ["onnx/*", "openvino/*", "coreml/*"]

report = {"models": {}, "git": {}, "hf_datasets": {}, "packages": {}}
for folder in (MODEL_DIR, DATA_DIR, WHEEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)
print("Python", sys.version.split()[0], "| torch", torch.__version__)

# %% [markdown]
# ## 2. Unduh model
#
# Bobot format `.bin` dikonversi ke safetensors supaya bisa dimuat di versi torch lokal yang
# lebih lama.

# %%
def to_safetensors(folder: Path) -> None:
    """Konversi pytorch_model.bin menjadi model.safetensors bila belum ada."""
    bin_file = folder / "pytorch_model.bin"
    target = folder / "model.safetensors"
    if not bin_file.exists():
        return
    if target.exists():
        bin_file.unlink()
        return
    state = torch.load(bin_file, map_location="cpu", weights_only=True)
    state = {k: v.clone().contiguous() for k, v in state.items() if isinstance(v, torch.Tensor)}
    save_file(state, target, metadata={"format": "pt"})
    bin_file.unlink()


for name, repo in MODELS.items():
    try:
        local = MODEL_DIR / name
        snapshot_download(
            repo, local_dir=local, allow_patterns=KEEP_PATTERNS, ignore_patterns=SKIP_PATTERNS
        )
        to_safetensors(local)
        shutil.rmtree(local / ".cache", ignore_errors=True)
        size_mb = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e6
        report["models"][name] = {"repo": repo, "status": "ok", "mb": round(size_mb)}
    except Exception as exc:
        report["models"][name] = {"repo": repo, "status": f"GAGAL: {exc}"[:300]}
    print(name, report["models"][name])

# %% [markdown]
# ## 3. Unduh data tambahan

# %%
for name, url in GIT_REPOS.items():
    target = DATA_DIR / name
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)], capture_output=True, text=True
    )
    shutil.rmtree(target / ".git", ignore_errors=True)
    report["git"][name] = "ok" if result.returncode == 0 else result.stderr[-300:]
    print(name, report["git"][name])

# %%
for name, (repo, config) in HF_DATASETS.items():
    try:
        dataset = load_dataset(repo, config)
        target = DATA_DIR / name
        target.mkdir(exist_ok=True)
        sizes = {}
        for split, part in dataset.items():
            part.to_pandas().to_parquet(target / f"{split}.parquet")
            sizes[split] = len(part)
        report["hf_datasets"][name] = {"repo": repo, "status": "ok", "rows": sizes}
    except Exception as exc:
        report["hf_datasets"][name] = {"repo": repo, "status": f"GAGAL: {exc}"[:300]}
    print(name, report["hf_datasets"][name])

# %% [markdown]
# ## 4. Unduh wheel library
#
# Hanya paket yang belum ada di image Kaggle yang diunduh. Wheel ini berlaku untuk Linux dan versi
# Python Kaggle. Untuk mesin lokal lain, lihat `requirements.txt` di folder ini.

# %%
dry_run = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", "--report", "pip_report.json"]
    + PACKAGES,
    capture_output=True,
    text=True,
)
needed = []
if Path("pip_report.json").exists():
    plan = json.loads(Path("pip_report.json").read_text())
    needed = [f"{i['metadata']['name']}=={i['metadata']['version']}" for i in plan["install"]]
    Path("pip_report.json").unlink()
report["packages"]["missing_on_kaggle"] = needed
report["packages"]["dry_run_error"] = dry_run.stderr[-300:] if dry_run.returncode else ""

for requirement in needed:
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--quiet", "--no-deps", "-d", str(WHEEL_DIR),
         requirement],
        check=False,
    )
report["packages"]["wheels"] = sorted(f.name for f in WHEEL_DIR.iterdir())
print(len(report["packages"]["wheels"]), "wheel:", report["packages"]["wheels"])

# %%
versions = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
).stdout
keep = ("torch", "transformers", "scikit-learn", "pandas", "numpy", "sentence-transformers",
        "umap-learn", "hdbscan", "rouge-score", "sacrebleu", "yake", "rank-bm25", "ruptures",
        "sentencepiece", "protobuf", "gensim", "jiwer", "lightgbm", "accelerate", "safetensors",
        "scipy", "matplotlib", "seaborn", "huggingface-hub", "tokenizers", "peft")
lines = [ln for ln in versions.splitlines() if ln.split("==")[0].lower() in keep]
(OUT_DIR / "requirements.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %% [markdown]
# ## 5. Laporan skema dataset
#
# Laporan ini menjadi dasar penulisan notebook task, supaya nama kolom yang dipakai benar-benar ada.

# %%
LABEL_HINTS = ("label", "sentim", "class", "target", "rating", "score", "emosi", "emotion", "hoax")


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip", encoding_errors="ignore")
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(suffix)


def describe_file(path: Path) -> list[str]:
    frame = read_any(path)
    lines = [f"\n### `{path}`", f"baris={len(frame)} kolom={list(frame.columns)}"]
    for col in frame.columns:
        lowered = str(col).lower()
        nunique = frame[col].astype(str).nunique()
        if any(h in lowered for h in LABEL_HINTS) or nunique <= 15:
            counts = frame[col].astype(str).value_counts().head(8).to_dict()
            lines.append(f"- {col}: nunique={nunique} {counts}")
    sample = frame.head(2).astype(str).map(lambda v: v[:150]).to_dict("records")
    lines.append(f"- contoh: {sample}")
    return lines


def describe_tree(root: Path, max_files: int = 60) -> list[str]:
    out: list[str] = []
    files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and f.suffix.lower() in {".csv", ".tsv", ".xlsx", ".json", ".jsonl", ".parquet"}
    ]
    for path in files[:max_files]:
        try:
            out += describe_file(path)
        except Exception as exc:
            out.append(f"\n### `{path}` GAGAL DIBACA: {exc}"[:300])
    return out


schema = ["# Laporan skema", json.dumps(report, indent=1)]
schema += ["\n## Input Kaggle"] + describe_tree(INPUT_DIR)
schema += ["\n## Data tambahan (HF)"] + describe_tree(DATA_DIR, max_files=30)
tree = sorted(str(f.relative_to(DATA_DIR)) for f in DATA_DIR.rglob("*") if f.is_file())
schema += ["\n## Daftar file data tambahan"] + tree[:300]
(OUT_DIR / "schema_report.md").write_text("\n".join(schema), encoding="utf-8")
print("Laporan skema:", len(schema), "baris")

# %% [markdown]
# ## 6. Manifest paket
#
# Notebook task mencari file `pack_manifest.json` untuk menemukan lokasi paket ini.

# %%
(OUT_DIR / "pack_manifest.json").write_text(json.dumps(report, indent=1))
total_gb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e9
print(f"Ukuran paket: {total_gb:.2f} GB")
print(json.dumps({k: v.get("status") for k, v in report["models"].items()}, indent=1))
print(json.dumps({k: v.get("status") for k, v in report["hf_datasets"].items()}, indent=1))
