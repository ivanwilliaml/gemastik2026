#!/usr/bin/env python3
"""
01_download_all.py  --  unduh SEMUA kebutuhan hari H (jalankan di laptop yang ADA internet).

Hanya pakai library bawaan Python 3.12 (tidak perlu pip install apa pun dulu).

Contoh:
  python 01_download_all.py --out D:\\comp\\bundle --check            # cek semua link/ID/ukuran, tanpa unduh
  python 01_download_all.py --out D:\\comp\\bundle --profile core     # unduh yang wajib
  python 01_download_all.py --out D:\\comp\\bundle --profile plus     # + model dan data tambahan (disarankan)
  python 01_download_all.py --out D:\\comp\\bundle --only F4,D1       # hanya item tertentu (ID dari tabel laporan)

Aman dijalankan ulang: file yang sudah lengkap dilewati, unduhan terputus dilanjutkan.
Kode item (A1, C2, dst) mengikuti tabel di laporan "Persiapan final NLP offline MBG.md".
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (gemastik-prep)"}
PROFILES = {"core": 0, "plus": 1, "all": 2}
TORCH_VER = "2.14.1"
PYPI = "https://pypi.org/simple"
TORCH_INDEX = "https://download.pytorch.org/whl/{v}"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ----------------------------------------------------------------------------- manifest
def item(id, kind, spec, dest, mb, prof, note, **kw):
    return dict(id=id, kind=kind, spec=spec, dest=dest, mb=mb, prof=prof, note=note, **kw)


IN = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/"
NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
LLAMA = "https://github.com/ggml-org/llama.cpp/releases/download/b12119/"
NUSAX = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/"
NUSAX_LANGS = ("acehnese", "balinese", "banjarese", "buginese", "english", "indonesian", "javanese", "madurese", "minangkabau", "ngaju", "sundanese", "toba_batak")
GEOB = "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/IDN/"

ITEMS = [
    # --- A. installer
    item("A1", "files", [("https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe", "installers/python-3.12.10-amd64.exe")], "", 27, "core", "Python 3.12.10"),
    item("A2", "files", [("https://aka.ms/vs/17/release/vc_redist.x64.exe", "installers/vc_redist.x64.exe")], "", 26, "core", "VC++ runtime"),
    # --- B. paket python (pip download)
    item("B2", "pipmain", None, "wheelhouse", 2700, "core", "pip/setuptools + torch cu130 + semua paket di requirements.in"),
    item("B5", "torchvar", "cu126", "wheelhouse_cu126", 2603, "core", "torch cu126: cadangan kalau driver PC panitia belum mendukung CUDA 13"),
    item("B3", "torchvar", "cpu", "wheelhouse_cpu", 130, "core", "torch CPU: cadangan terakhir"),
    # --- C. model inti
    item("C1", "hf", "intfloat/multilingual-e5-small", "models/e5-small", 471, "core", "embedding"),
    item("C2", "hf", "indolem/indobertweet-base-uncased", "models/indobertweet", 445, "core", "basis fine-tuning teks sosmed"),
    item("C3", "hf", "w11wo/indonesian-roberta-base-sentiment-classifier", "models/sent-w11wo", 499, "core", "sentimen siap pakai"),
    item("C4", "hf", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "models/mdeberta-nli", 558, "core", "zero-shot NLI"),
    item("C5", "hf", "cahya/NusaBert-ner-v1.3", "models/nusabert-ner", 644, "core", "NER"),
    # --- D. leksikon + data kecil
    item("D1", "files", [(("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"), "data_ext/lexicon/colloquial-indonesian-lexicon.csv")], "", 3, "core", "kamus slang"),
    item("D2", "files", [("https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/kamus_singkatan.csv", "data_ext/lexicon/kamus_singkatan.csv")], "", 0.02, "core", "singkatan"),
    item("D3", "files", [("https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv", "data_ext/lexicon/new_kamusalay.csv")], "", 0.3, "core", "kamus alay (baca latin-1)"),
    item("D4", "files", [(NLTK + "corpora/stopwords.zip", "nltk_data/corpora/stopwords.zip")], "", 0.04, "core", "stopword NLTK (HANYA untuk kata topik)", extract=True),
    item("D5", "files", [(NLTK + "tokenizers/punkt_tab.zip", "nltk_data/tokenizers/punkt_tab.zip")], "", 4.3, "core", "tokenizer NLTK", extract=True),
    item("D6", "files", [(IN + "smsa_doc-sentiment-prosa/train_preprocess.tsv", "data_ext/indonlu/smsa/train_preprocess.tsv"),
                         (IN + "smsa_doc-sentiment-prosa/valid_preprocess.tsv", "data_ext/indonlu/smsa/valid_preprocess.tsv")], "", 2.5, "core", "SmSA sentimen"),
    item("D7", "files", [(IN + "emot_emotion-twitter/train_preprocess.csv", "data_ext/indonlu/emot/train_preprocess.csv")], "", 0.7, "core", "EmoT"),
    item("D8", "files", [("https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv", "data_ext/hate/re_dataset.csv")], "", 1.9, "core", "hate speech multi-label (baca latin-1)"),
    item("D9", "ghfile", "fidopecado/Indonesian-Tiktok-Comments-Sentiment-Analysis|MBG_PROCESSED.csv", "data_ext/mbg/MBG_PROCESSED.csv", 20, "core", "komentar TikTok MBG, label otomatis (bukan gold)"),
    item("D10", "ghzip", "andikapratamazaid/analisis-program-mbg", "data_ext/mbg/analisis-program-mbg.zip", 10, "core", "fakta MBG untuk slide"),
    # --- E. dokumentasi offline
    item("E1", "files", [("https://scikit-learn.org/stable/_downloads/scikit-learn-docs.zip", "docs/scikit-learn-docs.zip")], "", 100, "core", "docs scikit-learn"),
    item("E2", "files", [("https://pandas.pydata.org/docs/pandas.zip", "docs/pandas.zip")], "", 34, "core", "docs pandas"),
    item("E3", "files", [("https://numpy.org/doc/stable/numpy-html.zip", "docs/numpy-html.zip"),
                         ("https://matplotlib.org/stable/Matplotlib.pdf", "docs/Matplotlib.pdf"),
                         ("https://www.python.org/ftp/python/doc/3.12.10/python-3.12.10-docs-html.zip", "docs/python-3.12.10-docs-html.zip")], "", 96, "plus", "docs numpy, matplotlib, python"),
    # --- F. model tambahan
    item("F1", "hf", "LazarusNLP/NusaBERT-base", "models/nusabert-base", 443, "plus", "basis Jawa/Sunda"),
    item("F2", "hf", "indobenchmark/indobert-base-p1", "models/indobert-base-p1", 498, "plus", "basis teks berita (hanya .bin, nanti dikonversi)"),
    item("F3", "hf", "Rifky/Indobert-QA", "models/indobert-qa", 440, "plus", "QA Indonesia"),
    item("F4", "hf", "LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta", "models/indonli-lite", 47, "plus", "NLI Indonesia"),
    item("F5", "hf", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "models/minilm-l12", 590, "plus", "embedding cadangan CPU", extra=["onnx/model_quint8_avx2.onnx"]),
    item("F6a", "hf", "crypter70/IndoBERT-Sentiment-Analysis", "models/sent-crypter70", 498, "plus", "sentimen pembanding"),
    item("F6b", "hf", "mdhugol/indonesia-bert-sentiment-classification", "models/sent-mdhugol", 498, "plus", "sentimen pembanding"),
    item("F7a", "hf", "Exqrch/IndoBERTweet-HateSpeech", "models/hate-exqrch", 442, "all", "hate speech"),
    item("F7b", "hf", "Exqrch/IndoBERTweet-Profanity", "models/profanity-exqrch", 442, "all", "profanity"),
    item("F8", "hf", "StevenLimcorn/indonesian-roberta-base-emotion-classifier", "models/emotion-steven", 499, "all", "emosi"),
    item("F9", "hf", "intfloat/multilingual-e5-base", "models/e5-base", 1112, "all", "embedding lebih besar"),
    item("F10", "hf", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "models/mpnet-multi", 1112, "all", "SetFit/parafrase"),
    item("F11", "hf", "panggi/t5-small-indonesian-summarization-cased", "models/t5-small-sum", 242, "all", "ringkasan"),
    item("F12", "hf", "w11wo/indonesian-roberta-base-posp-tagger", "models/pos-w11wo", 496, "all", "POS"),
    item("F13", "hf", "FacebookAI/xlm-roberta-base", "models/xlmr-base", 1116, "all", "basis multibahasa"),
    item("F14", "hf", "timpal0l/mdeberta-v3-base-squad2", "models/mdeberta-squad2", 1113, "all", "QA multibahasa"),
    item("F15", "hf", "cahya/t5-base-indonesian-summarization-cased", "models/t5-base-sum", 892, "all", "ringkasan besar"),
    # --- G. LLM lokal (hanya kalau aturan mengizinkan)
    item("G1", "files", [(LLAMA + "llama-b12119-bin-win-cuda-12.4-x64.zip", "llama/llama-b12119-bin-win-cuda-12.4-x64.zip"),
                         (LLAMA + "cudart-llama-bin-win-cuda-12.4-x64.zip", "llama/cudart-llama-bin-win-cuda-12.4-x64.zip")], "", 678, "all", "llama-server CUDA"),
    item("G2", "files", [(LLAMA + "llama-b12119-bin-win-vulkan-x64.zip", "llama/llama-b12119-bin-win-vulkan-x64.zip")], "", 38, "all", "llama-server Vulkan (cadangan)"),
    item("G3", "hf", "unsloth/gemma-4-E2B-it-GGUF", "gguf/gemma-4-E2B-it", 3107, "all", "LLM utama", only=["gemma-4-E2B-it-Q4_K_M.gguf"]),
    item("G5", "hf", "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF", "gguf/sea-lion-e2b", 3428, "all", "LLM pembanding", only=["Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf"]),
    item("G6", "hf", "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "gguf/qwen-coder-3b", 2105, "all", "asisten kode", only=["qwen2.5-coder-3b-instruct-q4_k_m.gguf"]),
    # --- H. data tambahan
    item("H1", "files", [(IN + "casa_absa-prosa/train_preprocess.csv", "data_ext/indonlu/casa/train_preprocess.csv"),
                         (IN + "hoasa_absa-airy/train_preprocess.csv", "data_ext/indonlu/hoasa/train_preprocess.csv")], "", 0.5, "plus", "ABSA"),
    item("H2", "files", [("https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/train0.csv", "data_ext/indolem/sentiment_train0.csv"),
                         ("https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/javanese/train.csv", "data_ext/nusax/javanese_train.csv"),
                         ("https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/sundanese/train.csv", "data_ext/nusax/sundanese_train.csv")], "", 0.6, "plus", "IndoLEM + NusaX"),
    item("H3", "files", [("https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl", "data_ext/indonli/train.jsonl")], "", 3.8, "plus", "IndoNLI"),
    item("H4", "files", [("https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv", "data_ext/inset/positive.tsv"),
                         ("https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv", "data_ext/inset/negative.tsv")], "", 0.13, "plus", "InSet (fitur saja, bukan pemberi label)"),
    item("H5", "files", [("https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1048/Emoji_Sentiment_Data_v1.0.csv", "data_ext/emoji/Emoji_Sentiment_Data_v1.0.csv")], "", 0.08, "plus", "sentimen emoji"),
    item("H6", "files", [("https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip", "data_ext/nrc/NRC-Emotion-Lexicon.zip")], "", 26, "all", "NRC EmoLex"),
    item("H7", "files", [("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz", "data_ext/fasttext/cc.id.300.vec.gz")], "", 1227, "all", "fastText vec (JANGAN .bin)"),
    item("H11", "hfdata", "tasyapndya/free_nutritious_meals_sentiment_x_id_2025", "data_ext/mbg/hf_x_sentiment", 5, "plus", "MBG X berlabel manual (~1.1k baris)"),
    item("F6c", "hf", "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis", "models/sent-aardiiiiy", 442, "plus", "sentimen pembanding, dipakai pada label TikTok MBG publik"),
    item("H12", "files", [(f"{NUSAX}sentiment/{lang}/{split}.csv", f"data_ext/nusax_full/sentiment/{lang}/{split}.csv") for lang in NUSAX_LANGS for split in ("train", "valid", "test")]
                         + [(f"{NUSAX}lexicon/{lang}.csv", f"data_ext/nusax_full/lexicon/{lang}.csv") for lang in NUSAX_LANGS if lang != "indonesian"]
                         + [(f"{NUSAX}mt/{split}.csv", f"data_ext/nusax_full/mt/{split}.csv") for split in ("train", "valid", "test")], "", 4, "plus", "NusaX lengkap: sentimen, padanan kata, terjemahan"),
    item("H13", "hfdata", "AksaraLLM/aksara-bahasa-daerah-v1", "data_ext/aksara", 80, "plus", "korpus Wikipedia 8 bahasa daerah, tanpa label"),
    item("H14", "files", [("https://raw.githubusercontent.com/yonatanyl/KODE-WILAYAH-KEPMENDAGRI-2025/main/KODE-WILAYAH-KEPMENDAGRI-2025.xlsx", "data_ext/wilayah/KODE-WILAYAH-KEPMENDAGRI-2025.xlsx")], "", 4, "plus", "kode wilayah Kepmendagri 2025, hasil OCR"),
    item("H15", "files", [(f"{GEOB}ADM1/geoBoundaries-IDN-ADM1_simplified.geojson", "data_ext/wilayah/idn_adm1.geojson"),
                          (f"{GEOB}ADM2/geoBoundaries-IDN-ADM2_simplified.geojson", "data_ext/wilayah/idn_adm2.geojson")], "", 25, "plus", "batas wilayah untuk peta slide, CC BY 3.0 IGO"),
    item("H16", "hfdata", "Abdoel03/MBG", "data_ext/mbg/hf_youtube", 1, "plus", "komentar YouTube MBG Jun-Jul 2026, tanpa label, untuk latihan"),
]

MANUAL = [
    "A3  Installer VS Code, Chrome (untuk kaleido), driver NVIDIA RTX 3070: unduh dari situs resmi, simpan di installers/.",
    "E4  Docs Transformers/sentence-transformers: git clone --depth 1, simpan folder docs/ (tidak ada zip resmi).",
    "E5  Zeal + docset PyTorch: https://github.com/zealdocs/zeal/releases (opsional).",
    "E6  VSIX VS Code (ms-python.python, pylance, debugpy, jupyter): lewat menu Download VSIX di marketplace.",
    "G8  Ekstensi Continue (VSIX win32-x64, rilis v1.2.23-vscode) kalau mau asisten kode offline.",
    "H8  MBG TikTok mentah (Kaggle sinryurifal/dataset-komentar-tiktok-mbg-makan-bergizi-gratis): butuh login Kaggle.",
    "H9  MBG YouTube: github.com/briannaufal211/mbg-youtube-sentiment-analysis, Kaggle dzikriraihan/indonesian-youtube-comments-mbg-program.",
    "H10 MBG Instagram: https://data.mendeley.com/datasets/gfsn8x62dk/1 (unduh lewat browser).",
    "CATATAN: pakai data MBG/eksternal hanya kalau panitia mengizinkan data eksternal.",
]


# ----------------------------------------------------------------------------- http helpers
def open_url(url, headers=None, method="GET", timeout=60):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def head_info(url):
    """(status, size or None). HEAD dulu, kalau ditolak coba GET Range 0-0."""
    for method, hdr in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            with open_url(url, hdr, method=method, timeout=30) as r:
                size = None
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr and cr.split("/")[-1].isdigit():
                    size = int(cr.split("/")[-1])
                elif r.headers.get("Content-Length", "").isdigit():
                    size = int(r.headers["Content-Length"])
                return r.status, size
        except urllib.error.HTTPError as e:
            if e.code in (403, 405, 400) and method == "HEAD":
                continue
            return e.code, None
        except Exception as e:  # noqa
            last = repr(e)[:80]
            if method == "HEAD":
                continue
            return last, None
    return "ERR", None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, size=None, sha=None, retries=6):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return "skip"
    part = dest.with_name(dest.name + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if size is not None and have == size:
                break
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with open_url(url, hdr, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                n, last = have, time.time()
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        n += len(b)
                        if total > 30e6 and time.time() - last > 5:
                            print(f"\r      {dest.name}: {n/1e6:,.0f}/{total/1e6:,.0f} MB", end="", flush=True)
                            last = time.time()
            if total > 30e6:
                print()
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:
                if size is not None and part.exists() and part.stat().st_size == size:
                    break
                part.unlink(missing_ok=True)
                last_err = e
            elif e.code in (401, 403, 404):
                raise
            else:
                last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        time.sleep(min(5 * attempt, 30))
    else:
        raise RuntimeError(f"gagal unduh {url}: {last_err!r}")
    got = part.stat().st_size
    if size is not None and got != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ukuran salah {dest.name}: {got} != {size}")
    if sha and sha256_of(part) != sha:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 salah {dest.name}")
    part.replace(dest)
    return "ok"


def get_json(url):
    with open_url(url, timeout=60) as r:
        return json.load(r), r.headers


# ----------------------------------------------------------------------------- Hugging Face
SKIP_DIRS = ("onnx/", "openvino/", "tf/", "flax/", "coreml/", "rust/", "tflite/")
SKIP_EXT = (".onnx", ".h5", ".msgpack", ".ot", ".tflite", ".mlmodel", ".mlpackage", ".gguf", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".safetensors.index.json.bak")


def hf_tree(kind, repo):
    url = f"https://huggingface.co/api/{kind}/{repo}/tree/main?recursive=true"
    out = []
    while url:
        data, headers = get_json(url)
        out += data
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = m.group(1) if m else None
    return [e for e in out if e.get("type") == "file"]


def select_model_files(files, only=None, extra=None):
    extra = set(extra or [])
    if only:
        return [f for f in files if f["path"] in set(only)]
    keep = []
    for f in files:
        p = f["path"]
        low = p.lower()
        if p in extra:
            keep.append(f)
            continue
        if low.startswith(SKIP_DIRS) or low.endswith(SKIP_EXT) or low == ".gitattributes":
            continue
        name = p.rsplit("/", 1)[-1].lower()
        if name.endswith(".bin") and not name.startswith("pytorch_model"):
            continue
        # sisa checkpoint training (optimizer 890 MB di indobertweet) tidak berguna untuk inference
        if name in ("optimizer.pt", "scheduler.pt", "scaler.pt") or name.startswith("rng_state"):
            continue
        if low.startswith(("checkpoint-", "runs/", "logs/")) or "/checkpoint-" in low:
            continue
        keep.append(f)
    # per folder: kalau ada safetensors, buang pytorch_model*.bin
    dirs_with_st = {f["path"].rsplit("/", 1)[0] if "/" in f["path"] else "" for f in keep if f["path"].endswith(".safetensors")}
    final = []
    for f in keep:
        d = f["path"].rsplit("/", 1)[0] if "/" in f["path"] else ""
        n = f["path"].rsplit("/", 1)[-1].lower()
        if d in dirs_with_st and n.startswith("pytorch_model"):
            continue
        final.append(f)
    return final


def hf_download(it, out, kind="models"):
    files = hf_tree(kind, it["spec"])
    if kind == "datasets":
        sel = [f for f in files if f["size"] <= 200e6 and not f["path"].startswith(".")]
    else:
        sel = select_model_files(files, it.get("only"), it.get("extra"))
    if not sel:
        raise RuntimeError("tidak ada file terpilih")
    base = f"https://huggingface.co/{'datasets/' if kind == 'datasets' else ''}{it['spec']}/resolve/main/"
    dest = out / it["dest"]
    for f in sel:
        sha = (f.get("lfs") or {}).get("oid")
        print(f"    - {f['path']} ({f['size']/1e6:,.1f} MB)")
        download(base + urllib.parse.quote(f["path"]), dest / f["path"], f["size"], sha)
    return sum(f["size"] for f in sel)


# ----------------------------------------------------------------------------- GitHub
def gh_find(repo, filename):
    data, _ = get_json(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
    hits = [e for e in data.get("tree", []) if e["path"].lower().endswith(filename.lower())]
    if not hits:
        raise RuntimeError(f"{filename} tidak ada di {repo}")
    return hits[0]


# ----------------------------------------------------------------------------- pip
def pip(*args, env_extra=None):
    env = os.environ.copy()
    env.update(env_extra or {})
    cmd = [sys.executable, "-m", "pip", *args]
    print("    $ " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"pip gagal ({r.returncode})")


def pip_main(out, cache):
    wh = out / "wheelhouse"
    env = {"PIP_CACHE_DIR": str(cache)}
    pip("download", "pip", "setuptools", "wheel", "-d", str(wh), "--only-binary=:all:", env_extra=env)
    pip("download", f"torch=={TORCH_VER}+cu130", "-d", str(wh), "--only-binary=:all:",
        "--index-url", TORCH_INDEX.format(v="cu130"), "--extra-index-url", PYPI, env_extra=env)
    pip("download", "-r", str(out / "tools" / "requirements.in"), "-c", str(out / "tools" / "constraints_cu130.txt"),
        "--find-links", str(wh), "-d", str(wh), "--only-binary=:all:", env_extra=env)
    torchs = list(wh.glob("torch-*.whl"))
    if len(torchs) != 1 or "cu130" not in torchs[0].name:
        raise RuntimeError(f"wheelhouse harus berisi tepat satu torch +cu130, ditemukan: {[t.name for t in torchs]}")


def pip_torchvar(out, variant, dest, cache):
    pip("download", f"torch=={TORCH_VER}+{variant}", "-d", str(out / dest), "--only-binary=:all:", "--no-deps",
        "--index-url", TORCH_INDEX.format(v=variant), "--extra-index-url", PYPI, env_extra={"PIP_CACHE_DIR": str(cache)})


# ----------------------------------------------------------------------------- run / check
def selected(args):
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        sel = [i for i in ITEMS if i["id"].upper() in want or re.sub(r"[a-z]$", "", i["id"]).upper() in want]
        missing = want - {i["id"].upper() for i in sel} - {re.sub(r"[a-z]$", "", i["id"]).upper() for i in sel}
        if missing:
            sys.exit(f"ID tidak dikenal: {sorted(missing)}")
        return sel
    rank = PROFILES[args.profile]
    skip = {x.strip().upper() for x in (args.skip or "").split(",") if x.strip()}
    return [i for i in ITEMS if PROFILES[i["prof"]] <= rank and i["id"].upper() not in skip]


def check_item(it):
    """Return (ok, mb_or_None, detail)."""
    k = it["kind"]
    try:
        if k == "files":
            total, bad = 0, []
            for url, rel in it["spec"]:
                st, size = head_info(url)
                if st not in (200, 206):
                    bad.append(f"{rel}: HTTP {st}")
                total += size or 0
            return (not bad), total / 1e6, "; ".join(bad) or f"{len(it['spec'])} file"
        if k in ("hf", "hfdata"):
            kind = "datasets" if k == "hfdata" else "models"
            files = hf_tree(kind, it["spec"])
            sel = [f for f in files if f["size"] <= 200e6] if k == "hfdata" else select_model_files(files, it.get("only"), it.get("extra"))
            miss = [p for p in (it.get("extra") or []) + (it.get("only") or []) if p not in {f["path"] for f in files}]
            if miss:
                return False, None, f"file tidak ada di repo: {miss}"
            return True, sum(f["size"] for f in sel) / 1e6, f"{len(sel)} file"
        if k == "ghfile":
            repo, fn = it["spec"].split("|")
            e = gh_find(repo, fn)
            return True, e.get("size", 0) / 1e6, e["path"]
        if k == "ghzip":
            st, size = head_info(f"https://github.com/{it['spec']}/archive/HEAD.zip")
            return st in (200, 206), (size or 0) / 1e6, f"HTTP {st}"
        if k in ("pipmain", "torchvar"):
            v = "cu130" if k == "pipmain" else it["spec"]
            with open_url(TORCH_INDEX.format(v=v) + "/torch/", timeout=60) as r:
                html = r.read().decode("utf-8", "replace")
            name = f"torch-{TORCH_VER}%2B{v}-cp312-cp312-win_amd64.whl"
            alt = name.replace("%2B", "+")
            if name in html or alt in html:
                return True, it["mb"], f"torch {TORCH_VER}+{v} cp312 win_amd64 ada di indeks PyTorch"
            return False, None, f"torch {TORCH_VER}+{v} cp312 win_amd64 TIDAK ada di indeks PyTorch"
    except Exception as e:  # noqa
        return False, None, repr(e)[:120]
    return False, None, "kind tidak dikenal"


def run_item(it, out, cache):
    k = it["kind"]
    if k == "files":
        for url, rel in it["spec"]:
            dest = out / rel
            print(f"    - {rel}")
            download(url, dest)
            if it.get("extract") and dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as z:
                    z.extractall(dest.parent)
    elif k == "hf":
        hf_download(it, out, "models")
    elif k == "hfdata":
        hf_download(it, out, "datasets")
    elif k == "ghfile":
        repo, fn = it["spec"].split("|")
        e = gh_find(repo, fn)
        print(f"    - {e['path']} ({e.get('size', 0)/1e6:,.1f} MB)")
        download(f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(e['path'])}", out / it["dest"], e.get("size"))
        head = (out / it["dest"]).read_bytes()[:60]
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError("file ini pointer Git LFS, bukan datanya; unduh manual lewat browser")
    elif k == "ghzip":
        download(f"https://github.com/{it['spec']}/archive/HEAD.zip", out / it["dest"])
    elif k == "pipmain":
        pip_main(out, cache)
    elif k == "torchvar":
        pip_torchvar(out, it["spec"], it["dest"], cache)


def copy_tools(out):
    tools = out / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("01_download_all.py", "02_install_offline.ps1", "02_install_offline.bat", "03_verify_offline.py",
                 "requirements.in", "constraints_cu130.txt", "constraints_cu126.txt", "constraints_cpu.txt", "README.md"):
        src = HERE / name
        if src.exists() and src.resolve() != (tools / name).resolve():
            shutil.copy2(src, tools / name)


def write_manifest(out):
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and "_pipcache" not in p.parts and p.name != "bundle_manifest.csv" and not p.name.endswith(".part"):
            rows.append((p.relative_to(out).as_posix(), p.stat().st_size))
    with open(out / "bundle_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "size"])
        w.writerows(rows)
    return len(rows), sum(s for _, s in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=r"D:\comp\bundle", help="folder bundle (nanti disalin ke SSD eksternal)")
    ap.add_argument("--profile", choices=PROFILES, default="core")
    ap.add_argument("--only", help="daftar ID dipisah koma, mis. F4,D1 (mengabaikan --profile)")
    ap.add_argument("--skip", help="ID yang dilewati, mis. G1,B5")
    ap.add_argument("--check", action="store_true", help="hanya cek ketersediaan dan ukuran, tanpa unduh")
    ap.add_argument("--yes", action="store_true", help="jangan tanya konfirmasi")
    args = ap.parse_args()

    out = Path(args.out)
    if sys.version_info[:2] != (3, 12):
        print(f"PERINGATAN: Python {sys.version.split()[0]}. Wheelhouse dibuat untuk cp312; jalankan dengan Python 3.12.")
        if not args.check:
            sys.exit(1)
    items = selected(args)
    print(f"Profil: {args.only and 'custom' or args.profile} | {len(items)} item | output: {out}\n")

    if args.check:
        bad = 0
        total = 0.0
        for it in items:
            ok, mb, detail = check_item(it)
            bad += not ok
            total += mb or it["mb"]
            size = f"{mb:,.0f} MB" if mb is not None else "?"
            print(f"  [{'OK  ' if ok else 'GAGAL'}] {it['id']:<4} {it['kind']:<8} {size:>9}  {it['spec'] if isinstance(it['spec'], str) else it['note']}  -> {detail}")
        print(f"\nTotal perkiraan: {total/1000:,.1f} GB | gagal: {bad}")
        sys.exit(1 if bad else 0)

    est = sum(i["mb"] for i in items)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    print(f"Perkiraan unduhan: {est/1000:,.1f} GB | ruang bebas di tujuan: {free:,.0f} GB")
    if free < est / 1000 * 1.3:
        sys.exit("Ruang disk tidak cukup (butuh perkiraan x1.3).")
    if not args.yes and input("Lanjut? [y/N] ").strip().lower() != "y":
        sys.exit("Dibatalkan.")

    copy_tools(out)  # requirements.in & constraints dipakai pip_main
    cache = out / "_pipcache"
    failed = []
    t0 = time.time()
    for it in items:
        print(f"\n[{it['id']}] {it['note']}  ({it['kind']})")
        try:
            run_item(it, out, cache)
            print("    selesai")
        except Exception as e:  # noqa
            print(f"    GAGAL: {e}")
            failed.append((it["id"], str(e)[:160]))

    shutil.rmtree(cache, ignore_errors=True)
    n, size = write_manifest(out)
    print(f"\n=== Selesai dalam {(time.time()-t0)/60:.0f} menit | {n} file, {size/1e9:,.1f} GB di {out}")
    if failed:
        print("ITEM GAGAL (jalankan ulang, atau pakai --only):")
        for i, e in failed:
            print(f"  {i}: {e}")
    print("\nYANG HARUS DIUNDUH MANUAL:")
    for m in MANUAL:
        print("  " + m)
    print(f"\nLangkah berikut: salin seluruh folder {out} ke SSD eksternal, lalu latihan dengan tools\\02_install_offline.bat (Wi-Fi mati).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
