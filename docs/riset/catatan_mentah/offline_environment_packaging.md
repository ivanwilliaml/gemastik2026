# Offline Python ML/NLP environment on Windows, and 500k rows of Indonesian text on weak hardware

Research date: 2026-11-05. Nothing was installed on the user's machine.

**How the evidence was gathered.** Three labels are used below.
- **Measured** means I queried the live index or server on 2026-11-05 (PyPI JSON, download.pytorch.org listings, Hugging Face API, HEAD requests).
- **Resolver-tested** means I ran `pip install --dry-run --ignore-installed --only-binary=:all: --report` twice on this machine (Python 3.12.6, pip 24.2). This installs nothing. It only proves that metadata and Windows wheels are consistent, not that the code works at runtime.
- **Estimate** means my own arithmetic or judgement, not a benchmark. No timing in these notes was measured on this laptop.

Two corrections to the brief:
- "GK" is not a GTX 1650 chip name. The notes treat the card as Turing, TU117 (sm_75).
- The current PyTorch is newer than the installed 2.12.1. The newest Windows wheel is 2.15.0.

## 1. CUDA PyTorch for the GTX 1650: wheel, index URL, size, AMP

### Takeaway
Recent PyTorch still supports this GPU. Use `torch==2.14.1+cu130` from `https://download.pytorch.org/whl/cu130` (1,991 MB wheel). Train in fp32 by default, never use bf16, and allow fp16 only after a NaN test on this actual laptop.

### Cited Findings
- **Wheels for cp312-win_amd64 (measured):**

| Index | Newest cp312 Windows wheel | Wheel size |
|---|---|---|
| cu126 | 2.14.1+cu126 | 2,603 MB |
| cu128 | 2.11.0+cu128 | 2,753 MB |
| cu130 | 2.14.1+cu130 | 1,991 MB |
| cu132 | 2.15.0+cu132 | 2,108 MB |
| cpu | 2.15.0+cpu | 127 MB |

  Source: [cu130 listing](https://download.pytorch.org/whl/cu130/torch/), [cu132 listing](https://download.pytorch.org/whl/cu132/torch/), [cu126 listing](https://download.pytorch.org/whl/cu126/torch/).
- CUDA 13.x wheels cover "Turing (7.5) through Blackwell". PyTorch 2.14 is the last release with CUDA 12.x wheels, and Maxwell, Pascal and Volta are dropped afterwards — [PyTorch dev-discuss notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).
- NVIDIA's cuFFT docs say the minimum architecture for CUDA 13 is Turing — [NVIDIA deprecated functionality](https://docs.nvidia.com/cuda/cufft/deprecated-functionality.html).
- As of release 2.12, PyTorch labelled CUDA 13.0 "Stable" and 13.2 "Experimental". That table was for Linux x86_64 — [PyTorch dev-discuss, CUDA 13.2](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337).
- NVIDIA no longer lists a Windows minimum driver for CUDA 13.0+. The Linux floor for 13.2 is 595.45.04, and the Windows column is N/A — [CUDA 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/), [NVIDIA forum thread](https://forums.developer.nvidia.com/t/compatible-nvidia-gpu-drivers-on-windows-for-cuda-toolkit-13-0/371618).
- The GTX 16 series is Turing with no Tensor Cores. The desktop GTX 1650 has 896 CUDA cores and 4 GB of GDDR5 or GDDR6 — [NVIDIA 16-series comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/).
- fp16 NaNs are documented on GTX 1660, 1660 Ti and 1660 Super (TU116), while fp32 was fine. The issue was closed as fixed by cuDNN 8.2.2, but users still reported NaNs on later versions up to torch 2.0.1, and in July 2023 a commenter called it unsolved on 16xx cards. The thread does not mention the GTX 1650 — [pytorch/pytorch#58123](https://github.com/pytorch/pytorch/issues/58123).
- A forum user on a GTX 1650 Ti saw memory roughly halve under autocast with no speedup — [PyTorch forum](https://discuss.pytorch.org/t/no-speedup-with-autocast/148273).
- bf16: Turing has no native bf16 path, and native support arrived with Ampere — [blog on T4 bf16](https://earino.github.io/applied-deep-learning/site/blog/05_bf16_on_t4.html). `torch.cuda.is_bf16_supported()` is reported inconsistently: the [PyTorch forum](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629) says it requires Ampere+, while the same blog says it returned True on a Turing T4. Do not rely on the helper.

### Inferences
- **Why cu130 rather than cu132.** Both should work on Turing. cu130 is a patch release on the line PyTorch called stable, and its wheel is smaller. Measured, the Windows 2.15.0 wheel exists only on cu132, so choosing 2.15 means choosing 13.2. If 2.14.1+cu130 fails the verify script, try 2.15.0+cu132, then 2.14.1+cu126.
- **Driver.** 610.91 is numerically well above the 595.x Linux floor, but numbering differs across platforms, so this is only suggestive. The real test is `nvidia-smi` showing CUDA 13.x and `torch.cuda.is_available()` returning True. Do not change the driver in the final days.
- **No CUDA Toolkit install is needed.** The wheel carries its own CUDA and cuDNN DLLs ([ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) state this about PyTorch).
- **fp16 risk for this task may be lower than the thread suggests.** Many reports in that thread involve image models. Whether text transformers on TU117 are affected is unknown, so test it. Since there is no speed gain to expect, fp16 is only a memory lever.

### fp16 self-test (run on the laptop, with a real model and real text)
```python
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel
path = r"D:\comp\models\indobert-base-p1"
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True).cuda().eval()
texts = ["Makan bergizi gratis di sekolah sangat membantu orang tua."] * 16
batch = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
with torch.inference_mode():
    a = model(**batch).last_hidden_state[:, 0].float().cpu()
    with torch.autocast("cuda", dtype=torch.float16):
        b = model(**batch).last_hidden_state[:, 0].float().cpu()
print("nan fp16:", torch.isnan(b).any().item())
print("min cosine fp32 vs fp16:", torch.cosine_similarity(a, b).min().item())
print("compute capability:", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
```
Pass criteria (my own thresholds): no NaN, and cosine above 0.999. For training, also run 200 steps with `GradScaler` and watch for a NaN loss or a scale that keeps collapsing. If either fails, stay on fp32.

### Gaps
- No source tests the GTX 1650 (TU117) specifically for fp16 NaNs.
- The NVIDIA spec page covers desktop cards. The laptop variant's clocks and GDDR type are not confirmed.
- The [get-started page](https://pytorch.org/get-started/locally/) fetched without readable selector content, so the "current stable" status comes from the index listings rather than that page. The cu131 and cu133 listings returned 403, which I read as nonexistent but cannot confirm.
- Windows 10 support in the CUDA 13.x system requirements was not confirmed from NVIDIA.

## 2. Offline package strategy: which approach is most robust

### Takeaway
Use a **wheelhouse plus a pinned lock file, installed with `pip --no-index --find-links`**, and prove it with a fresh venv with Wi-Fi off. This is also what makes it portable to a competition-provided PC. A copied venv is only a fast path on the same machine. uv is optional, and conda-pack adds nothing here.

### Cited Findings
- `pip download` flags `--only-binary=:all:` and `--dest` build a wheelhouse. Cross-target flags require `--only-binary=:all:` or `--no-deps` — [pip download docs](https://pip.pypa.io/en/stable/cli/pip_download/).
- uv has `--offline`, `--no-index` and `--find-links` — [uv CLI reference](https://docs.astral.sh/uv/reference/cli/). `uv pip compile` writes a lockfile, and `uv pip sync` makes an environment match it exactly — [uv compile docs](https://docs.astral.sh/uv/pip/compile/).
- `uv venv --relocatable` avoids hardcoding the absolute path in activation scripts and writes relative console-script entry points on Windows — [uv source](https://docs.rs/uv-virtualenv/0.0.55/src/uv_virtualenv/virtualenv.rs.html), [origin commit](https://app.semanticdiff.com/gh/astral-sh/uv/commit/cb47aed9dea568811c6c5dd526574eb3cae739b0).
- PyTorch's WinError 126 on `c10.dll` or `fbgemm.dll` is usually a missing DLL dependency, typically the VC++ runtime, and one user's missing file was libomp140 — [PyTorch forum](https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969).
- XGBoost also requires the Visual C++ Redistributable DLLs — [XGBoost install docs](https://xgboost.readthedocs.io/en/stable/install.html).
- Installers checked (measured): [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) 25.6 MB, [python-3.12.10-amd64.exe](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) 27.0 MB, [embeddable zip](https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip) 11.1 MB.

### Inferences: ranking
| Approach | Same laptop | Other PC | Verdict |
|---|---|---|---|
| Wheelhouse + lock + `--no-index` | Yes | Yes, with Python 3.12 and VC++ | **Primary** |
| Copy of a built venv | Yes, at the same path | Fragile, since `pyvenv.cfg` points at a specific Python | Keep as a second copy on the SSD |
| uv relocatable venv | Yes | Better, but uv is new to this setup and untested here | Optional |
| conda-pack | Requires installing conda | Would work | Skip: no conda present, and it adds a new toolchain the day before |

### Runbook (PowerShell, text only, for the user to run)
Use online time **well before** the eve if possible. The torch wheel alone is 2 GB.

**A. Layout and installers**
```powershell
New-Item -ItemType Directory -Force D:\comp\wheelhouse, D:\comp\models, D:\comp\docs, D:\comp\installers, D:\comp\nltk_data, D:\comp\code, D:\comp\vsix
# Put in D:\comp\installers: python-3.12.10-amd64.exe, vc_redist.x64.exe, VS Code installer, Chrome installer, current NVIDIA driver installer
```

**B. Build the wheelhouse (online)**
```powershell
py -3.12 -m venv D:\comp\venv_build
D:\comp\venv_build\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip download pip setuptools wheel -d D:\comp\wheelhouse

# 1. Torch from the CUDA index
python -m pip download "torch==2.14.1+cu130" -d D:\comp\wheelhouse --only-binary=:all: `
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple

# 2. Everything else, forced to reuse that torch
Set-Content D:\comp\constraints.txt "torch==2.14.1+cu130"
python -m pip download -r D:\comp\requirements.in -c D:\comp\constraints.txt `
  --find-links D:\comp\wheelhouse -d D:\comp\wheelhouse --only-binary=:all:

# 3. Sdist-only pure-Python extras (optional; builds locally, not tested by me)
python -m pip wheel --no-deps flashtext compress-fasttext -w D:\comp\wheelhouse
```
Without the constraint, pip will pull PyPI's torch, whose Windows wheel measured only 124 MB, which indicates a CPU build. That is the exact trap that gave the current "+cpu" state. Check afterwards that only one `torch-*.whl` is in the folder.

**C. Lock and rehearse (Wi-Fi OFF, and unplug LAN)**
```powershell
py -3.12 -m venv D:\comp\env
D:\comp\env\Scripts\python -m pip install --no-index --find-links D:\comp\wheelhouse --upgrade pip
D:\comp\env\Scripts\python -m pip install --no-index --find-links D:\comp\wheelhouse -r D:\comp\requirements.in -c D:\comp\constraints.txt
D:\comp\env\Scripts\python -m pip freeze > D:\comp\requirements.lock
D:\comp\env\Scripts\python -m pip check
```
Then delete `D:\comp\env`, reinstall from `requirements.lock` with the same `--no-index` flags, and run the verify script in section 3. A rehearsal only counts if it passes with the network off **after a reboot**.

**D. Moving to another PC**
```powershell
python-3.12.10-amd64.exe /quiet InstallAllUsers=0 PrependPath=0 TargetDir=C:\comp\py312
vc_redist.x64.exe /install /quiet /norestart
C:\comp\py312\python -m venv C:\comp\env
C:\comp\env\Scripts\python -m pip install --no-index --find-links E:\comp\wheelhouse -r E:\comp\requirements.lock
```

### Gaps
- `requirements.lock` is not yet real, because it comes from the user's first install. The resolver test used PyPI's torch 2.14.1 metadata rather than the +cu130 wheel. A mismatch is unlikely but only step C proves it.
- Python 3.12.10's installer flags and the VC++ `/install /quiet` flags are from my own knowledge, not fetched in this session. Run them `/?` first if something fails.
- libomp140 is the one known DLL not covered by the VC++ redistributable in the forum report. If a clean PC shows WinError 126 after installing the redist, this is the first suspect.

## 3. Hugging Face offline: downloads, env vars, pitfalls

### Takeaway
Download with `--local-dir` into plain folders and load by **path**. This sidesteps symlinks, cache hashes and most offline failures. Set `HF_HUB_OFFLINE=1` before any import. Do not download whole repos: for several models the full repo is 2 to 10 times the model you need.

### Cited Findings
- `HF_HUB_OFFLINE=1` blocks all HTTP calls to the Hub and raises an error if a file is not cached. All variables are read at import time of `huggingface_hub`, so they must be set before importing — [HF env var docs](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
- The current Transformers offline docs describe `HF_HUB_OFFLINE=1` and `local_files_only=True` and do not list `TRANSFORMERS_OFFLINE` — [Transformers install docs](https://huggingface.co/docs/transformers/installation). The Datasets docs likewise name `HF_HUB_OFFLINE` for full offline mode — [Datasets load docs](https://huggingface.co/docs/datasets/loading). Those pages rendered at datasets v4.8.4 docs versus the installed-candidate 5.1.0.
- Windows symlinks: without Developer Mode or admin, the cache stores files directly in `snapshots/`, which works but can duplicate files. `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences the warning — [HF cache docs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
- If a snapshot is incomplete while offline, `snapshot_download` raises `IncompleteSnapshotError` instead of silently returning a partial folder. `hf cache verify` checks checksums — [HF cache docs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
- `hf_xet` is used automatically when present, and `HF_HUB_DISABLE_XET` turns it off. `hf_transfer` is deprecated. `HF_HUB_DISABLE_UPDATE_CHECK=1` stops the `hf` CLI's PyPI check, and `HF_HUB_DISABLE_TELEMETRY=1` stops telemetry — [HF env var docs](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
- `SENTENCE_TRANSFORMERS_HOME` or `cache_folder` set the model location. `local_files_only` restricts loading to local files. `trust_remote_code` defaults to False — [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html).
- NLTK: `python -m nltk.downloader -d <dir> ...` plus `NLTK_DATA` pointing at the top folder — [NLTK data docs](https://www.nltk.org/data.html). Measured: [punkt_tab.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip) 4.3 MB, and stopwords.zip reachable.
- `evaluate.load()` can fail offline for Hub-hosted metrics, and it accepts a local script path. `HF_EVALUATE_OFFLINE=1` is only reported by a user — [evaluate loading docs](https://huggingface.co/docs/evaluate/package_reference/loading_methods), [evaluate#315](https://github.com/huggingface/evaluate/issues/315), [seqeval discussion](https://huggingface.co/spaces/evaluate-metric/seqeval/discussions/3).
- **Model files (measured via the [HF API](https://huggingface.co/api/models/indobenchmark/indobert-base-p1?blobs=true), 2026-11-05):**

| Repo | Whole repo | Weights | Format | Tokenizer files |
|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 977 MB | 91 MB | safetensors, bin | tokenizer.json, vocab.txt |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 4,621 MB | 471 MB | safetensors, bin | tokenizer.json, sentencepiece |
| intfloat/multilingual-e5-small | 2,280 MB | 471 MB | safetensors, bin | tokenizer.json, sentencepiece |
| LazarusNLP/all-indobert-base-v4 | 499 MB | 498 MB | safetensors | tokenizer.json, vocab.txt |
| indobenchmark/indobert-base-p1 | 1,652 MB | 498 MB | **bin only** | **vocab.txt only** |
| indolem/indobert-base-uncased | 887 MB | 445 MB | **bin only** | **vocab.txt only** |
| cahya/distilbert-base-indonesian | 273 MB | 273 MB | **bin only** | **vocab.txt only** |
| distilbert/distilbert-base-multilingual-cased | 2,906 MB | 542 MB | safetensors, bin | tokenizer.json, vocab.txt |
| w11wo/indonesian-roberta-base-sentiment-classifier | 1,499 MB | 499 MB | safetensors, bin | tokenizer.json |
| mdhugol/indonesia-bert-sentiment-classification | 498 MB | 498 MB | **bin only** | **vocab.txt only** |
| cahya/bert-base-indonesian-NER | 883 MB | 443 MB | **bin only** | **vocab.txt only** |
| BAAI/bge-m3 | 4,587 MB | 2,271 MB | bin (root) | tokenizer.json, sentencepiece |

  None of the 12 reported `auto_map` in config, so `trust_remote_code` should not be needed for any of them. None is gated.
- FastText Indonesian vectors (measured): [cc.id.300.bin.gz](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz) 4,507 MB and [cc.id.300.vec.gz](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz) 1,227 MB.

### Inferences: runbook
**env.ps1** (dot-source it in every shell; never rely on `setx` alone):
```powershell
$env:HF_HOME="D:\comp\hf"; $env:SENTENCE_TRANSFORMERS_HOME="D:\comp\hf\st"; $env:NLTK_DATA="D:\comp\nltk_data"
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:HF_DATASETS_OFFLINE="1"; $env:HF_EVALUATE_OFFLINE="1"
$env:HF_HUB_DISABLE_TELEMETRY="1"; $env:DO_NOT_TRACK="1"; $env:HF_HUB_DISABLE_UPDATE_CHECK="1"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
$env:PYTORCH_ALLOC_CONF="expandable_segments:True"; $env:TOKENIZERS_PARALLELISM="false"
```
`TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` are belt-and-braces: harmless if ignored. In a notebook, set the same variables in the first cell **before any import**.

**Download (online, with `HF_HUB_OFFLINE` unset):**
```powershell
$inc = @("*.json","*.txt","*.model","model.safetensors","pytorch_model.bin")
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --include $inc --local-dir D:\comp\models\mlm-l12
hf download indobenchmark/indobert-base-p1 --include $inc --local-dir D:\comp\models\indobert-base-p1
python -m nltk.downloader -d D:\comp\nltk_data punkt_tab stopwords
```
If a Xet download stalls, retry with `$env:HF_HUB_DISABLE_XET="1"`.

**Convert bin-only models now, while online.** Six of the 12 candidates ship only `pytorch_model.bin` and `vocab.txt`. Load each once, then:
```python
model.save_pretrained(r"D:\comp\models\indobert-st", safe_serialization=True)
tok.save_pretrained(r"D:\comp\models\indobert-st")   # also writes tokenizer.json
```
This removes both the pickle-loading path and the slow-to-fast tokenizer conversion on the day.

**Load by path only:**
```python
SentenceTransformer(r"D:\comp\models\mlm-l12", local_files_only=True, device="cuda")
AutoModel.from_pretrained(r"D:\comp\models\indobert-st", local_files_only=True)
```

**Other rules**
- Use `sklearn.metrics` instead of `evaluate.load`. It removes a class of offline failures.
- Avoid `trust_remote_code=True` entirely, because the code can pull extra files.
- SSD portability: flat folders survive exFAT and copying. A symlinked HF cache may not.

**verify_offline.py** (Wi-Fi off; the socket guard makes any hidden network call fail loudly):
```python
import socket
def _blocked(*a, **k): raise RuntimeError("NETWORK CALL ATTEMPTED")
socket.socket.connect = _blocked

import importlib, torch
mods = """pandas polars pyarrow numpy scipy sklearn lightgbm xgboost catboost transformers datasets accelerate
tokenizers sentence_transformers peft evaluate setfit bertopic umap hdbscan gensim onnxruntime optimum cleanlab
optuna matplotlib seaborn plotly wordcloud networkx jupyterlab ipykernel ipywidgets tqdm joblib ahocorasick
rapidfuzz datasketch emoji openpyxl pptx kaleido nltk sentencepiece tiktoken""".split()
for m in mods:
    try: importlib.import_module(m); print("ok  ", m)
    except Exception as e: print("FAIL", m, repr(e)[:200])

print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.rand(2000, 2000, device="cuda"); print("gpu matmul", (x @ x).sum().item())

import onnxruntime as ort; print(ort.get_available_providers())
import xgboost as xgb, numpy as np
d = xgb.DMatrix(np.random.rand(5000, 20), label=np.random.randint(0, 2, 5000))
xgb.train({"device": "cuda", "tree_method": "hist", "objective": "binary:logistic"}, d, 10); print("xgb gpu ok")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(r"D:\comp\models\mlm-l12", local_files_only=True, device="cuda")
print(m.encode(["makan bergizi gratis"]).shape)

import nltk; print(len(nltk.corpus.stopwords.words("indonesian")))
import plotly.express as px; px.bar(x=[1, 2], y=[1, 2]).write_image(r"D:\comp\kaleido_test.png"); print("kaleido ok")
```
Also run the fp16 self-test, a 50-step training loop, one `save_pretrained` and a `lightgbm` plus `catboost` GPU fit. Import failures in an optional package are acceptable only if you decide in advance to drop it.

### Gaps
- Exact `hf download` `--include` behaviour with an array was not run by me. Check that each folder contains `config.json`, tokenizer files and weights, and for sentence-transformers `modules.json` and `1_Pooling/config.json`.
- Whether Transformers 5.22.1 still loads `pytorch_model.bin` without conditions was not checked. That is why the conversion step exists.
- Whether `nltk` stopwords includes "indonesian" is from my memory. The verify script tests it.
- `hf cache verify` applies to the cache. I did not confirm a checksum path for `--local-dir` folders.

## 4. Package list: purpose, Python 3.12 Windows status, disk

### Takeaway
All of the requested packages have Python 3.12 Windows wheels, except `fasttext`, `flashtext` and `compress-fasttext`. A list of 52 top-level packages **resolved to 206 wheels with no conflict (resolver-tested)**, totalling about 2.7 GB of wheels with CUDA torch. Estimate 8 to 11 GB once installed. Runtime risk sits in a few old packages, mainly `fasttext-wheel` under NumPy 2 and `snorkel`.

### Cited Findings
Versions and wheel sizes are measured from [PyPI JSON](https://pypi.org/pypi/transformers/json) on 2026-11-05, and "OK" means a wheel exists for cp312/abi3/py3 on win_amd64. All were part of the resolver run unless noted.

| Package | Version | Wheel MB | 3.12 Win | Purpose |
|---|---|---|---|---|
| pandas | 3.1.1 | 10.8 | OK | DataFrame work |
| polars + polars-runtime-32 | 2.1.0 | 0.9 + 53.9 | OK | Fast CSV/parquet, lazy queries on 500k rows |
| pyarrow | 26.0.1 | 29.1 | OK | Parquet, Arrow strings |
| numpy | 2.5.4 | 12.6 | OK (needs >=3.12) | Arrays, memmap |
| scipy | 1.18.1 | 36.7 | OK | Sparse matrices |
| scikit-learn | 1.9.1 | 8.3 | OK | TF-IDF, hashing, SGD, LinearSVC, metrics |
| lightgbm | 4.7.0 | 1.4 | OK | GBDT on engineered features |
| xgboost | 3.4.1 | 48.9 | OK | GBDT with CUDA |
| catboost | 1.2.10 | 100.2 | OK | GBDT with categoricals (large, optional) |
| transformers | 5.22.1 | 13.1 | OK | Models |
| datasets | 5.1.0 | 0.6 | OK | Arrow-backed memory-mapped data |
| accelerate | 1.15.0 | 0.4 | OK | Trainer backend |
| tokenizers | 0.23.3 | 2.9 | OK | Fast tokenization |
| sentence-transformers | 6.2.0 | 0.8 | OK | Embeddings |
| peft | 0.21.2 | 0.8 | OK | LoRA and adapters (optional) |
| evaluate | 0.4.6 | 0.1 | OK | Needed by setfit; do not use for metrics offline |
| setfit | 1.2.0 | 0.1 | OK | Few-label classification |
| bertopic | 0.17.4 | 0.2 | OK | Topic modelling |
| umap-learn + pynndescent | 0.5.12 + 0.7.0 | 0.2 | OK | Dimensionality reduction |
| numba + llvmlite | 0.68.1 + 0.50.0 | 2.8 + 41.9 | OK | UMAP dependencies |
| hdbscan | 0.8.44 | 1.9 | OK | Clustering |
| fasttext-wheel | 0.9.2 | 0.2 | OK, but see below | Supervised fastText |
| gensim | 4.4.0 | 24.4 | OK | Word2Vec, loading `.vec`, FastText alternative |
| onnxruntime-gpu | 1.31.1 | 168.1 | OK | Fast inference (optional) |
| optimum | 2.3.0 | 0.2 | OK | ONNX export helper (optional) |
| cleanlab | 2.9.0 | 0.3 | OK | Label noise detection |
| snorkel | 0.10.0 (Feb 2024) | 0.1 | Resolves, runtime unknown | Weak supervision |
| optuna | 5.0.0 | 0.4 | OK | Tuning |
| matplotlib, seaborn | 3.11.2, 0.13.2 | 9.3, 0.3 | OK | Charts |
| plotly, kaleido | 7.1.0, 1.5.0 | 9.7, 0.1 | OK, needs Chrome | Interactive charts and static export |
| wordcloud | 1.9.6 | 0.3 | OK | Word clouds |
| networkx | 3.7 | 2.1 | OK | Graphs |
| jupyterlab, ipykernel, ipywidgets | 4.6.5, 7.4.0, 8.1.9 | 17.3, 0.1, 0.1 | OK | Notebooks |
| tqdm, joblib | 4.70.1, 1.6.0 | 0.1, 0.3 | OK | Progress, caching |
| pyahocorasick | 2.3.1 | small | OK | Fast keyword matching |
| flashtext | 2.7 (2018) | sdist | **No wheel** | Keyword matching; prefer pyahocorasick |
| rapidfuzz | 3.14.6 | 1.7 | OK | Fuzzy matching, normalizing slang |
| datasketch | 2.0.0 | 0.1 | OK | MinHash near-duplicates |
| emoji | 2.16.0 | 0.6 | OK | Emoji handling |
| openpyxl, python-pptx | 3.1.5, 1.0.2 | 0.3, 0.5 | OK | Excel, slides |
| compress-fasttext | 0.1.5 (2024) | sdist | **No wheel** | Compress fastText; drop it |
| fasttext | 0.9.3 | sdist | **No wheel** | Needs a compiler; do not use |

Additions I recommend, all resolver-tested: sentencepiece 0.2.2, tiktoken 0.14.0, protobuf 7.36.2, hf-xet 1.8.0, huggingface-hub 2.4.0, safetensors 0.8.0, nltk 3.10.4, psutil 7.2.2, nvidia-ml-py 13.615.71, Sastrawi 1.0.1 (Indonesian stemmer, from 2016).

- **The polars trap:** `polars` is a 0.9 MB shell depending on `polars-runtime-32` — [polars PyPI JSON](https://pypi.org/pypi/polars/json). A hand-copied wheel list would miss the runtime. `pip download` handles it.
- **fasttext under NumPy 2:** `predict` calls `np.array(probs, copy=False)`, which raises `ValueError` under NumPy 2. Fixes are editing `FastText.py` to use `np.asarray(probs)` or pinning NumPy below 2. The upstream repo was archived on 19 March 2024 — [HF forum thread](https://discuss.huggingface.co/t/valueerror-unable-to-avoid-copy-while-creating-an-array-as-requested/93584), [fasttext-language-identification discussion](https://huggingface.co/facebook/fasttext-language-identification/discussions/9). The resolver chose NumPy 2.5.4, so this bug applies by default.
- **fastText Windows builds:** sdist installs fail for lack of a compiler toolchain — [txtai#756](https://github.com/neuml/txtai/issues/756). `fasttext-wheel` has a cp312 win_amd64 wheel, uploaded 2024-02-17 (measured, [PyPI](https://pypi.org/pypi/fasttext-wheel/json)).
- **xgboost:** the default Windows wheel includes GPU and is built with CUDA 13.x. `xgboost-cu12` exists for older drivers, and `xgboost-cpu` is the small variant — [XGBoost docs](https://xgboost.readthedocs.io/en/stable/install.html).
- **lightgbm:** "The CUDA version is not supported on Windows". Use `device_type=gpu` (OpenCL) — [LightGBM install guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html).
- **onnxruntime-gpu:** its table lists CUDA 13.x with cuDNN 9.x for ORT 1.27 to 1.30, so 1.31.1 is one release beyond what the page lists. Importing torch first or calling `onnxruntime.preload_dlls()` reuses PyTorch's CUDA DLLs — [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- **kaleido:** v1 no longer bundles Chrome. It looks for a local install, and `plotly_get_chrome` or `kaleido_get_chrome` needs internet — [Kaleido PyPI](https://pypi.org/project/kaleido/), [Plotly static export](https://plotly.com/python/static-image-export/).

**Disk (wheel figures measured, installed figures are estimates):**
| Item | Size |
|---|---|
| All wheels except torch | about 0.69 GB (815 MB run total minus 124 MB PyPI torch) |
| torch 2.14.1+cu130 wheel | 1.99 GB |
| Wheelhouse total | **about 2.7 GB** |
| Installed venv (estimate) | 8 to 11 GB |
| Models, 3 to 5 picks with `--include` | 1.5 to 3 GB |
| `cc.id.300.vec.gz` (optional) | 1.2 GB |
| Docs zips (section 5) | about 0.25 GB |
| Installers, VSIX (estimate) | 0.5 to 1 GB |
| **SSD package total (estimate)** | **15 to 20 GB** |

### Inferences
- **What "resolver-tested" does not cover:** runtime compatibility. It included snorkel and `fasttext-wheel` but not the sdist-only trio. Snorkel 0.10.0 predates NumPy 2 and pandas 3, so a clean resolve does not mean it works. Writing labelling functions by hand with `cleanlab` is a safer default if snorkel fails the import or a small test. Its dependency tensorboard also adds weight.
- **pandas 3 is a major version.** Older snippets may behave differently, so rehearse your own code on it. If a snippet breaks, fix it during rehearsal rather than downgrading the day before.
- **Recommended cuts for simplicity:** flashtext, compress-fasttext and possibly peft and optimum.
- **fastText path:** use `fasttext-wheel` for training only, with the one-line patch tested during rehearsal, or use `gensim.models.KeyedVectors.load_word2vec_format(path, limit=200000)`. The full `.bin` is 4.5 GB compressed, so it is unrealistic in 8 GB of RAM.
- **Do not install both `onnxruntime` and `onnxruntime-gpu`.** The resolved list has only the GPU one. From memory, they conflict in the same environment.

### Gaps
- LightGBM's docs did not say whether the Windows pip wheel includes OpenCL GPU support. CatBoost GPU on the GTX 1650 was not checked. Test both, and treat the CPU as acceptable since both are quick on tabular features.
- `optimum` ONNX export may need extra packages, and `onnx` was not in the resolved list. If you want ONNX, test the full export path while online.
- Unpacked sizes were not measured.

## 5. Offline docs and tools

### Takeaway
Download official zips for scikit-learn, NumPy, pandas, Python and Matplotlib. Use Zeal for PyTorch. Transformers has no official zip that I could find, so keep the source Markdown from its repo. VS Code extensions go in via `.vsix`.

### Cited Findings
- Measured with HEAD requests:
  - [scikit-learn-docs.zip](https://scikit-learn.org/stable/_downloads/scikit-learn-docs.zip) 99.8 MB
  - [numpy-html.zip](https://numpy.org/doc/stable/numpy-html.zip) 36.5 MB
  - [pandas.zip](https://pandas.pydata.org/docs/pandas.zip) 34.1 MB
  - [Matplotlib.pdf](https://matplotlib.org/stable/Matplotlib.pdf) 46.5 MB
  - [python-3.12.10-docs-html.zip](https://www.python.org/ftp/python/doc/3.12.10/python-3.12.10-docs-html.zip) 13.2 MB
  - The guessed SciPy zip URL returned 404.
- Zeal's latest release is [v0.9.2](https://github.com/zealdocs/zeal/releases/tag/v0.9.2) (measured). Docsets download inside its Docset Library — [Zeal README](https://github.com/zealdocs/zeal). A University in Prague page lists prebuilt PyTorch 2.10, torchvision and scikit-learn docsets — [UFAL](https://ufal.mff.cuni.cz/node/1701).
- VS Code: right-click an extension, then **Download VSIX** or **Download Specific Version VSIX**. Install with `code --install-extension myextension.vsix`. Auto-update is disabled for VSIX installs — [VS Code docs](https://code.visualstudio.com/docs/configure/extensions/extension-marketplace).

### Inferences
- **Transformers, sentence-transformers:** `git clone --depth 1` their repos and keep `docs/`. `help()` and `?` in Jupyter work offline too. Also save your own one-page cheat sheet of tested snippets, which is more valuable under time pressure than reference docs.
- **VSIX set:** `ms-python.python`, `ms-python.vscode-pylance`, `ms-python.debugpy`, `ms-toolsai.jupyter` and its bundled dependents. Download them from the same VS Code version you will use, then test on an isolated `--extensions-dir` with Wi-Fi off. If VS Code has trouble, JupyterLab from the venv is the fallback, so never make VS Code a single point of failure.
- **Plotly in notebooks:** avoid CDN-based renderers such as `notebook_connected`. Test charts with Wi-Fi off.
- **Fonts:** if slides need a specific font, download and install it the day before. For wordcloud, pass an explicit `font_path` to a local `.ttf`.

### Gaps
- No official offline zip for PyTorch or Transformers docs was found. The [community PyTorch.docs repo](https://github.com/unknownue/PyTorch.docs) exists but its currency was not checked.
- DevDocs offline mode was not verified.
- Exact extension dependency lists and whether Pylance's VSIX is platform-specific were not checked.

## 6. 500k rows on 8 GB of RAM and 4 GB of VRAM

### Takeaway
Memory arithmetic favours you if you keep text out of Python objects, keep embeddings on disk and fine-tune on a **sample** of 30k to 100k rows. A complete pass over 500k is realistic for hashing or TF-IDF linear models and for small-model embedding, but not for fine-tuning.

### Cited Findings
- Out-of-core: stream, extract features, learn incrementally. `HashingVectorizer` is the stateless option, and `SGDClassifier`, `MultinomialNB` and `MiniBatchKMeans` among others have `partial_fit`. Pass all `classes=` on the first `partial_fit`. Vectorization often costs more than learning — [scikit-learn scaling docs](https://scikit-learn.org/stable/computing/scaling_strategies.html).
- Windows DataLoader: `spawn` is the default, so most of the script must be under `if __name__ == '__main__':`. `collate_fn` and dataset code must be top-level and picklable, and lambdas are not valid for `worker_init_fn`. Worker memory can approach workers × parent size for Python objects, so use NumPy, pandas or Arrow storage — [PyTorch data docs](https://docs.pytorch.org/docs/2.15/data.html).
- `Dataset.from_file()` memory-maps an Arrow file — [Datasets docs](https://huggingface.co/docs/datasets/loading).
- `PYTORCH_ALLOC_CONF` replaces the older `PYTORCH_CUDA_ALLOC_CONF` alias. `expandable_segments` (experimental) targets varying batch sizes, which is exactly dynamic padding. Overusing pinned memory "can cause serious problems when running low on RAM" — [PyTorch CUDA notes](https://docs.pytorch.org/docs/2.15/notes/cuda.html).
- Page file: commit limit = RAM + page files, and reaching it can cause freezing or crashes. System-managed files grow up to 3× RAM or 4 GB, capped at one-eighth of the volume — [Microsoft page file docs](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file).
- `Trainer` builds a length-grouped sampler with `group_by_length` — [HF forum](https://discuss.huggingface.co/t/how-to-implement-trainers-group-by-length-in-pytorch/9232). I could not confirm its name or defaults in v5.

### Inferences: memory arithmetic (estimates)
| Object | Calculation | RAM or disk |
|---|---|---|
| Embeddings, 384-d float32 | 500k × 384 × 4 B | 768 MB |
| Embeddings, 768-d float32 | 500k × 768 × 4 B | 1.54 GB |
| TF-IDF CSR, 30 terms per row | 15M nnz × 12 B | about 180 MB |
| TF-IDF CSR, 1-2 grams at 60 per row | 30M nnz × 12 B | about 360 MB |
| Raw text at 300 B, Arrow | 150 MB + offsets | about 155 MB |
| Same as Python strings | adds about 50 B per object | 200+ MB, and it multiplies across workers |

**Training VRAM rule:** weights + gradients + 2 Adam states = **4 × file size**, before activations.
| Model | File | 4× | Room for activations (of about 3.5 GB usable) |
|---|---|---|---|
| all-MiniLM-L6 | 91 MB | 0.36 GB | Plenty |
| distilbert-indonesian | 273 MB | 1.1 GB | Good |
| indobert-base | 498 MB | 2.0 GB | Tight |
| multilingual MiniLM-L12 | 471 MB | **1.9 GB** | Tight |

The non-obvious result: **the multilingual MiniLM-L12 costs about as much to fine-tune as a full BERT**, because its large multilingual vocabulary embedding dominates the file. It is cheap for inference speed but not for training. Freezing the embedding layer saves much of this. SetFit on it may be tight for the same reason.

**Timing estimates (mine, with an uncertainty of about ×2 either way; short social-media-length text assumed):**
| Task on 500k rows unless noted | Settings | Estimate |
|---|---|---|
| Read CSV, write parquet (polars) | 150 MB of text | under 1 min |
| Clean/normalize, pure Python regex | | 2 to 10 min |
| `HashingVectorizer` 2^20, 1-2 grams | single core | 1 to 4 min |
| TF-IDF word 1-2 grams, `min_df=5` | | 2 to 5 min |
| TF-IDF char 3-5 grams | RAM-heavy; use a sample | 5 to 20 min, 2 to 5 GB |
| `SGDClassifier`, 3 epochs | hashed sparse | 1 to 3 min |
| `LinearSVC(dual=False)` binary | TF-IDF | 3 to 15 min, multiply by classes for OvR |
| Embed, MiniLM-L6, batch 128, 128 tokens | fp32 | 5 to 15 min |
| Embed, multilingual MiniLM-L12, batch 64 to 128 | fp32 | 10 to 30 min |
| Embed, IndoBERT, batch 32 to 64, 128 tokens | fp32 | 30 to 80 min |
| Fine-tune IndoBERT, batch 8, 128 tokens | 50k rows, 1 epoch | 10 to 25 min |
| Fine-tune IndoBERT, batch 4, 256 tokens | 50k rows, 1 epoch | 25 to 60 min |
| Fine-tune DistilBERT, batch 16, 128 tokens | 50k rows, 1 epoch | 5 to 12 min |
| Fine-tune on all 500k rows | | 1.5 to 4 h per epoch: **do not do this** |
| UMAP + HDBSCAN on all 500k | | Likely beyond 8 GB and 1 h; use 20k to 50k, then assign the rest |

**Calibrate before trusting any of this:** time 10k rows for each step on real data, then multiply by 50. Batch sizes above are starting points: double until out of memory, then back off by 25%.

**Budget for a 5-hour day (suggested):** 30 min data and EDA; a TF-IDF plus linear baseline with a saved submission inside the first hour; 30 min to embed; fine-tune only one model on a sample; stop all training at T minus 60 min.

**Tactics**
- *Data:* convert to parquet first and never re-read the CSV. Use `pl.scan_parquet(...)` with lazy filters. Use `category` dtype for repeated columns and `float32` for numerics. Deduplicate early, as social data often has many duplicates, and this may be your largest speedup. Use `del df; gc.collect()` between stages.
- *Embeddings:* sort by length, encode in chunks and write to a memmap with resumable progress:
  ```python
  arr = np.lib.format.open_memmap(r"D:\comp\emb.npy", mode="w+", dtype="float32", shape=(n, 384))
  for start in range(0, n, 20000):
      arr[start:start+20000] = model.encode(texts[start:start+20000], batch_size=96, convert_to_numpy=True)
      arr.flush()
      open(r"D:\comp\emb.progress", "w").write(str(start + 20000))
  ```
  Remember to invert the length sort at the end.
- *Linear models:* use `HashingVectorizer(n_features=2**20, alternate_sign=False)` with `SGDClassifier(loss="log_loss")` and `partial_fit` first. Only then try TF-IDF plus `LinearSVC`.
- *Training on 4 GB:* dynamic padding with `DataCollatorWithPadding`, length bucketing, `max_length=128` first, gradient accumulation to reach an effective batch of 32, and gradient checkpointing only if needed (from general practice it trades speed for memory). Close anything using the GPU and check `nvidia-smi` first.
- *Workers:* start with `num_workers=0`, especially in notebooks. Fast tokenizers plus pre-tokenized Arrow data rarely need workers. Put long jobs in `.py` files with a main guard. From memory rather than from a source fetched here, each Windows worker re-imports torch and commits a lot of virtual memory, which can show up as WinError 1455 "paging file too small". On 8 GB, this makes 0 to 2 workers the sensible range.
- *Thread limits:* for `joblib` and sklearn, use `n_jobs=4` rather than 12 to limit memory and heat. Each process copies data on Windows.

**Page file recommendation.** Yes, increase it. It is a normal Windows setting, and the competition rules are the only thing that could object, which is unlikely since it is a local setting. Set a custom size on the fastest **SSD** at 32 GB initial and 32 GB maximum, or leave C: system-managed and add one on D: if D: is an SSD. It prevents commit-limit crashes, but it does not make swapping fast. If the task is actively paging, you are already losing, so reduce the data. Check drive types first with `Get-PhysicalDisk | Select FriendlyName,MediaType`. If D: or E: is an HDD, keep the venv, models and page file on the SSD where possible, as C: has 33 GB free and the working set is about 15 GB.

### Gaps
- No trustworthy public benchmark for GTX 1650 transformer throughput was found. All timings above are estimates.
- Argument names in `TrainingArguments` for v5 (such as `group_by_length`) are unverified. Check them with `help(TrainingArguments)` during rehearsal, or write a small custom bucketing sampler.
- `encode_multi_process` behaviour on Windows was not confirmed. It is not needed with one GPU.
- Drive types (SSD or HDD) on this laptop are unknown.

## 7. Reliability: power, thermals, updates, background runs, checkpoints

### Takeaway
Treat a reboot, a sleep, a kernel crash and thermal throttling as expected events. Run long jobs as separate processes with logs and checkpoints. Install pending Windows updates days before, not on the eve. Avoid starting a new undervolt close to the competition.

### Cited Findings
- `powercfg /change standby-timeout-ac`, `monitor-timeout-ac`, `hibernate-timeout-ac` take minutes as the value, and `/hibernate off` disables hibernation — [Microsoft powercfg](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/powercfg-command-line-options).
- Consumer Windows 10 ESU end date conflicts: [Microsoft Q&A](https://learn.microsoft.com/en-ca/answers/questions/5805066/windows-esu-does-enrollment-end-on-13-oct-2026-or) quotes 13 October 2026, while [TechRepublic](https://www.techrepublic.com/article/news-windows-10-esu-cheat-sheet/) and [SC World](https://www.scworld.com/brief/microsoft-extends-free-windows-10-security-updates-for-consumers) report an extension to 12 October 2027. I did not confirm this on Microsoft's own lifecycle page.
- Non-ESU Windows 10 PCs reportedly cannot use Pause Updates. This rests on one outlet with no Microsoft statement — [Windows Latest](https://windowslatest.com/2025/12/13/non-esu-windows-10-cant-pause-updates-anymore-including-if-windows-11-install-starts-accidentally-by-user).
- Thermals (all anecdotal owner reports, some on sister models with other GPUs):
  - A 15-dk0 with i7-9750H hits 100 °C and downclocks under gaming load — [HP forum](https://h30434.www3.hp.com/t5/Gaming-Notebooks/HP-Pavillion-Gaming-Laptop-15-dk0xxx-reaching-100c-and/m-p/8252431).
  - One owner reached 100 °C after 20 minutes despite an undervolt — [HP forum](https://h30434.www3.hp.com/t5/Notebook-Hardware-and-Upgrade-Questions/HP-pavillion-gaming-15-dk-extremely-high-cpu-and-gpu/m-p/8642296).
  - An undervolt of -0.155 V crashed in one game while -0.150 V held, in a search result from the [TechPowerUp ThrottleStop threads](https://www.techpowerup.com/forums/threads/my-throttlestop-settings-i7-9750h.312846). I saw these as search summaries and did not open the thread, so the exact post is not confirmed.
  - A cooling pad failed to stop throttling for one owner under stress tests, and a repaste gave no improvement for another. Both came from the same forum search summary, and I did not trace them to specific threads, so treat them as weak evidence.
  - On a sibling model with an i7-10750H, an owner reports HP locked undervolting — [HP forum](https://h30434.www3.hp.com/t5/Gaming-Notebooks/HP-Pavilion-15-dk1019na-Power-Limit-Exceeded/td-p/7978074).

### Inferences: runbook
**Power (run as admin, the day before):**
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /hibernate off
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
```
Use the Windows Power slider or High Performance plan. The `LIDACTION` alias is from memory, so check it with `powercfg /aliases`.

**Updates**
- About 3 to 7 days before, install all pending updates, reboot and rehearse after them.
- On the day, the network is off, so downloads cannot start, but an already-staged update can still force a restart. Check for "Restart required" the night before. Set Active Hours to cover the event.
- If you want a hard block, run as admin: `sc.exe stop wuauserv; sc.exe config wuauserv start= disabled`. Undo afterwards with `sc.exe config wuauserv start= demand`.

**Thermals, in order of value and safety**
1. Clean vents and raise the rear for airflow. A pad helps airflow but is not a cure.
2. Limit threads (`n_jobs=4`, `torch.set_num_threads(6)`) since GPU and CPU share heat here.
3. Optionally disable turbo with `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99`, then `/setactive SCHEME_CURRENT`. It trades peak speed for steadier temperatures, so test both on a real 30-minute job.
4. Undervolting only if you have weeks to test stability, as a crash costs more than the heat. Some HP models lock it.
5. Do not repaste days before the event.

Monitor during rehearsal:
```powershell
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks_throttle_reasons.active,power.draw,memory.used --format=csv -l 30 > D:\comp\gpu_log.csv
```
A 1-hour combined load during rehearsal tells you the true sustained speed, and the timing estimates in section 6 assume no throttling.

**Charger.** Bring the original HP adapter. Run on AC with a plug test at the venue if possible, plus an extension cord. Battery-only operation will throttle heavily and will not last 5 hours under load (general knowledge).

**Long jobs survive a kernel crash** when they are separate processes:
```powershell
Start-Process -FilePath D:\comp\env\Scripts\python.exe -ArgumentList "D:\comp\code\train.py","--run","r1" `
  -RedirectStandardOutput D:\comp\runs\r1.out -RedirectStandardError D:\comp\runs\r1.err -WindowStyle Hidden -PassThru
Get-Content D:\comp\runs\r1.out -Wait -Tail 20
```
Save the printed PID. Stop with `Stop-Process -Id <pid>`. Windows has no tmux equivalent by default, but a second PowerShell window is enough.

**Checkpointing:**
- `save_strategy="steps"`, `save_steps` about every 10 to 15 minutes of work, `save_total_limit=2` and `resume_from_checkpoint=True`.
- With Adam state each checkpoint is about 3× the model file, so keep them on D:.
- In custom loops, write to `x.tmp` then `os.replace` so a crash cannot leave a half-written file.
- Save intermediate outputs (parquet, memmap, `joblib.dump`) after every stage, along with a "last known good" submission.

**Backups:**
- Keep two copies of the package: on the laptop and on an external SSD.
- Verify with `Get-FileHash` on the torch wheel and a model file.
- Back up working code to the SSD each hour, if the rules allow an external drive.
- Keep the wheelhouse on the SSD even when installed, to allow a rebuild in about 10 to 20 minutes (estimate).

### Gaps
- Sustained thermal behaviour of this exact unit (GTX 1650 and 9750H) is unknown, as most reports are on GTX 1660 Ti siblings or games.
- The adapter wattage for this SKU was not verified.
- Whether Defender exclusions are worthwhile was not researched. Real-time scanning can slow pip installs, so if you add an exclusion for `D:\comp`, remove it afterwards.

## 8. RAM upgrade: can the 15-dk0xxx take 16/32 GB?

### Takeaway
Yes, by consistent HP community answers: **2 SODIMM slots, up to 2 × 16 GB = 32 GB, DDR4 at 2666 MHz**. Buy a matched pair if you can. This is the largest cheap gain for this task, and should be done before the wheelhouse rehearsal, not on the eve. The HP service guide itself was not retrieved.

### Cited Findings
- Two slots, each up to a 16 GB DDR4-2666 module — [HP community](https://h30434.www3.hp.com/t5/Notebook-Hardware-and-Upgrade-Questions/What-is-the-maximum-amount-of-RAM-that-can-be-installed-on/td-p/9208347), [HP community](https://h30434.www3.hp.com/t5/Gaming-Notebooks/Maximum-RAM-for-Upgrade/m-p/8034147).
- DDR4-3200 works but runs at 2666 MHz — [HP community](https://h30434.www3.hp.com/t5/Notebook-Hardware-and-Upgrade-Questions/Upgrade-RAM-in-notebook/m-p/8598392). One thread says DDR4-2933 for some 15-dk0 models — [HP community](https://h30434.www3.hp.com/t5/Notebook-Hardware-and-Upgrade-Questions/HP-Pavilion-Gaming-15-dk0xxx-RAM-compatibility-question-s/m-p/8087942) (fetch blocked; search snippet only).
- Kingston's listing shows a 32 GB maximum (snippet only; page returned 403) — [Kingston](https://kingston.com/tw/memory/search/model/100281/hp-hpe-pavilion-gaming-notebook-15-dk0xxx-15t-dk0xxx).
- Conflict: Crucial's scanner and one forum poster claim 2 × 32 GB (64 GB), contradicting HP-recommended answers — [HP community](https://h30434.www3.hp.com/t5/Gaming-Notebooks/Maximum-amount-of-RAM/m-p/7907439).
- Chapter 1 of the Maintenance and Service Guide has the memory specs and chapter 5 the removal procedure — [HP community](https://h30434.www3.hp.com/t5/Notebook-Software-and-How-To-Questions/MEMORIA-RAM/m-p/7946637).
- Forum replies say matching capacities are needed for dual channel, and that mixing 16 GB with 8 GB typically drops to single channel — [HP community](https://h30434.www3.hp.com/t5/Gaming-Notebooks/RAM-Upgrade/m-p/9544043/highlight/true).
- Attribution caveat: HP's forum blocked direct fetches (403), so every HP community point in this section comes from search-result summaries. The links are the threads those summaries returned, and the mapping of a given sentence to a given thread is best-effort. The 2 × 16 GB and 2666 MHz answer was consistent across results.

### Inferences
- **Check first (read-only):**
  ```powershell
  Get-CimInstance Win32_PhysicalMemory | Select BankLabel,Capacity,Speed,Manufacturer,PartNumber
  ```
  One 8 GB stick means one free slot. Then adding a matched 8 GB makes 16 GB in dual channel at the lowest cost. Two 16 GB is the sensible target (replacing whatever is in there). I would not buy 64 GB based on a scanner claim.
- **What it changes:** 16 GB removes most paging risk, and 32 GB makes char TF-IDF, BERTopic on a larger sample and multiple workers comfortable. It does not help VRAM or GPU training speed, as 4 GB VRAM is unchanged.
- **Timing:** Do it before rehearsal so all later testing happens on the final hardware. Boot and run a memory test (Windows Memory Diagnostic) afterwards. A bad stick found during the competition would be the worst outcome.
- **Rules and warranty:** this is a laptop of 2019 vintage, so warranty is probably moot, but check. Only the user can buy this, and it depends on local price, which I did not research.

### Gaps
- HP's own service guide text was not retrieved.
- The current slot population on this unit was not read.
- Indonesian market prices were not researched.

## 9. If the competition provides its own computer: portable package

### Takeaway
Ask the organizer now. If you cannot confirm that you may use your own laptop, prepare a portable SSD package that assumes **Windows, unknown GPU, no admin rights and possibly no Python**. The wheelhouse design handles this, except for a few cases that need a separate fallback.

### Cited Findings
- The installers and wheels in sections 1 and 2 are all measured as available. The CPU wheel is small (2.15.0+cpu at 127 MB; 2.14.1+cpu is also listed) — [cpu listing](https://download.pytorch.org/whl/cpu/torch/).
- CUDA 13.x wheels do not support GPUs below Turing, and cu126 is the last line that covers them, through 2.14 — [PyTorch notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).

### Inferences
**SSD contents**
```
E:\comp\
  installers\   python-3.12.10-amd64.exe, vc_redist.x64.exe, VS Code, Chrome, NVIDIA driver
  wheelhouse\   all wheels + torch +cu130
  wheelhouse_alt\  torch 2.14.1+cpu (always), torch 2.14.1+cu126 (2.6 GB, only if an old-GPU PC is likely)
  models\       flat folders loaded by path
  nltk_data\  docs\  vsix\  code\
  requirements.in  constraints.txt  requirements.lock
  env.ps1  verify_offline.py  fp16_test.py  README_SETUP.txt
```
**Decision tree on a new PC**
1. `python --version`, `nvidia-smi`, RAM and free disk.
2. If Python is not 3.12, per-user install 3.12.10 into a new folder. NumPy 2.5.4 needs >=3.12 and all cp312 wheels need exactly 3.12.
3. NVIDIA and Turing or newer: use the same lock. Pre-Turing: swap in cu126. No NVIDIA: swap in +cpu, using a separate constraint file, and rely on TF-IDF plus linear models and small ONNX or CPU embedding on a sample.
4. VC++ missing and no admin rights: this is the hardest case. Ask the organizer ahead of time. A possible workaround is placing the redistributable DLLs beside the Python executable or in torch's `lib`, which I have not tested.
5. Run `verify_offline.py`, and adjust paths via `env.ps1`.

**Rehearse this on a second Windows machine** if you can borrow one. It is the only real proof, since your own laptop already holds runtimes and caches that can hide a missing dependency.

**Also ask the organizer:** whether external drives and pre-downloaded models are allowed, whether pretrained weights are permitted, and power socket availability.

### Gaps
- Competition rules are unknown to me.
- The no-admin VC++ workaround is unverified.
- `torch 2.14.1+cpu` wheel size was not separately measured, though it was listed.
