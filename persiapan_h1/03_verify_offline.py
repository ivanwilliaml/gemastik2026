#!/usr/bin/env python3
"""
03_verify_offline.py -- cek environment OFFLINE dan UKUR kecepatan nyata di mesin ini.

Dijalankan otomatis oleh 02_install_offline.ps1, atau manual:
    %COMP_ROOT%\\env\\Scripts\\python 03_verify_offline.py [--no-bench]

Semua panggilan jaringan diblokir; kalau ada library yang diam-diam butuh internet, ia gagal dengan jelas.
Hasil benchmark (teks/detik, langkah/detik) menggantikan semua estimasi waktu di laporan.
Exit code 0 = semua komponen penting lolos.
"""
import importlib
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("COMP_ROOT") or Path(__file__).resolve().parent.parent)
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[k] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

NO_BENCH = "--no-bench" in sys.argv
problems = []  # masalah yang membuat exit code != 0


def blocked(*a, **k):
    raise RuntimeError("NETWORK CALL ATTEMPTED")


socket.socket.connect = blocked


def section(t):
    print(f"\n=== {t}")


# ---------------------------------------------------------------- import paket
section("Import paket")
ESSENTIAL = """pandas pyarrow numpy scipy sklearn lightgbm xgboost transformers datasets accelerate tokenizers
sentence_transformers setfit bertopic umap hdbscan gensim optuna matplotlib seaborn plotly networkx jupyterlab
ipykernel tqdm joblib psutil openpyxl nltk sentencepiece safetensors huggingface_hub""".split()
OPTIONAL = """polars evaluate cleanlab onnxruntime wordcloud ipywidgets ahocorasick rapidfuzz datasketch emoji
fast_langdetect Sastrawi pptx kaleido tiktoken pynvml""".split()
for m in ESSENTIAL + OPTIONAL:
    try:
        importlib.import_module(m)
        print("ok   ", m)
    except Exception as e:  # noqa
        tag = "FAIL " if m in ESSENTIAL else "warn "
        print(tag, m, repr(e)[:160])
        if m in ESSENTIAL:
            problems.append(f"import {m}")

import numpy as np  # noqa: E402

try:
    import psutil  # noqa: E402
except Exception:  # noqa
    psutil = None

try:
    import torch  # noqa: E402
except Exception as e:  # noqa
    print("FAIL torch", repr(e)[:200])
    sys.exit(1)

# ---------------------------------------------------------------- mesin
section("Mesin")
if psutil:
    vm = psutil.virtual_memory()
    print(f"Python {sys.version.split()[0]} | RAM {vm.total/1e9:.1f} GB (bebas {vm.available/1e9:.1f}) | CPU {psutil.cpu_count(logical=False)}c/{psutil.cpu_count()}t")
else:
    print(f"Python {sys.version.split()[0]} | psutil tidak ada, info RAM dilewati")
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
cuda = torch.cuda.is_available()
dev = "cuda" if cuda else "cpu"
bf16 = False
if cuda:
    p = torch.cuda.get_device_properties(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU {p.name} | VRAM {p.total_memory/1e9:.1f} GB | compute {p.major}.{p.minor} | bf16: {bf16}")
    x = torch.rand(2000, 2000, device="cuda")
    print("gpu matmul ok:", float((x @ x).sum()) > 0)
else:
    print("PERINGATAN: GPU tidak terbaca -> semua akan berjalan di CPU")
    problems.append("GPU tidak terbaca")

# ---------------------------------------------------------------- NLTK
section("NLTK")
try:
    import nltk
    from nltk.corpus import stopwords
    print("stopword indonesian:", len(stopwords.words("indonesian")))
    nltk.data.find("tokenizers/punkt_tab")
    print("punkt_tab ok")
except Exception as e:  # noqa
    print("warn NLTK:", repr(e)[:160])

# ---------------------------------------------------------------- model
M = ROOT / "models"
TEXTS = [
    "Program makan bergizi gratis sangat membantu anak sekolah di daerah kami.",
    "Menu MBG hari ini nasi, ayam goreng, sayur, dan buah, anak-anak lahap makannya.",
    "Kok banyak kasus keracunan ya, tolong BGN evaluasi dapur SPPG yang bermasalah!",
    "Anggaran MBG triliunan rupiah, apakah sudah tepat sasaran dan transparan?",
    "Alhamdulillah UMKM dan petani lokal ikut kebagian pasokan bahan makanan.",
    "gk ngerti deh kenapa susunya sering telat, padahal programnya bagus bgt.",
]
texts = (TEXTS * 400)[:2000]


def bench_embed():
    from sentence_transformers import SentenceTransformer
    path = M / "e5-small"
    if not path.exists():
        print("SKIP e5-small tidak ada")
        return
    m = SentenceTransformer(str(path), local_files_only=True, device=dev)
    if cuda:
        m.half()
    m.max_seq_length = 64
    print("dimensi embedding:", m.encode(["passage: makan bergizi gratis"]).shape)
    m.encode(texts[:256], batch_size=256)
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    m.encode(texts, batch_size=256, show_progress_bar=False)
    if cuda:
        torch.cuda.synchronize()
    r = len(texts) / (time.time() - t)
    print(f"[{dev}] e5-small embedding: {r:,.0f} teks/detik -> 500.000 baris ~ {500000/r/60:.1f} menit (diukur, max_len 64)")


def bench_bert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = M / "indobertweet"
    if not path.exists():
        print("SKIP indobertweet tidak ada")
        return
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True, num_labels=3).to(dev)
    # 1) kewarasan presisi: fp16/bf16 vs fp32 pada encoder
    batch = tok(texts[:16], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
    model.eval()
    with torch.inference_mode():
        a = model.base_model(**batch).last_hidden_state[:, 0].float().cpu()
        if cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
                b = model.base_model(**batch).last_hidden_state[:, 0].float().cpu()
            print(f"presisi campuran: nan={bool(torch.isnan(b).any())} min cosine vs fp32={torch.cosine_similarity(a, b).min().item():.5f}")
            if torch.isnan(b).any():
                problems.append("NaN pada presisi campuran")
    # 2) kecepatan inferensi
    enc = tok(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    bs = 128

    def infer():
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                kw = {k: v[i:i + bs].to(dev) for k, v in enc.items()}
                if cuda:
                    with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
                        model(**kw)
                else:
                    model(**kw)
    infer()
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    infer()
    if cuda:
        torch.cuda.synchronize()
    r = len(texts) / (time.time() - t)
    print(f"[{dev}] IndoBERTweet inferensi: {r:,.0f} teks/detik -> 500.000 baris ~ {500000/r/60:.1f} menit (diukur, max_len 64)")
    # 3) kecepatan fine-tuning
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    bs_t = 32
    y = torch.randint(0, 3, (bs_t,), device=dev)
    kw = {k: v[:bs_t].to(dev) for k, v in enc.items()}
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and not bf16)

    def step():
        opt.zero_grad(set_to_none=True)
        if cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
                loss = model(**kw, labels=y).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = model(**kw, labels=y).loss
            loss.backward()
            opt.step()
        return loss

    for _ in range(5):
        step()
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    n = 20
    for _ in range(n):
        loss = step()
    if cuda:
        torch.cuda.synchronize()
    sps = n * bs_t / (time.time() - t)
    print(f"[{dev}] fine-tuning IndoBERTweet (bs {bs_t}, len 64): {sps:,.0f} baris/detik -> 100.000 baris/epoch ~ {100000/sps/60:.1f} menit; loss={float(loss):.3f}")
    if cuda:
        print(f"VRAM puncak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    if not np.isfinite(float(loss)):
        problems.append("loss tidak finite saat fine-tuning")


if NO_BENCH:
    print("\n(benchmark dilewati)")
else:
    for name, fn in (("Embedding (e5-small)", bench_embed), ("IndoBERTweet", bench_bert)):
        section(name)
        try:
            fn()
        except Exception as e:  # noqa
            print("FAIL", repr(e)[:300])
            problems.append(name)

section("Ringkasan")
if problems:
    print("MASALAH:", "; ".join(problems))
    sys.exit(1)
print("Semua komponen penting lolos.")
