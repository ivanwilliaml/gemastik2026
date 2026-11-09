"""Uji semua snippet dan resep pada data lokal.

Setiap snippet dieksekusi di namespace kosong (ditambah file `needs`-nya), sehingga import yang
terlupa langsung gagal. Tes berjalan di namespace terpisah.

Pakai:
  python tools/test_snippets.py                         semua snippet, tanpa model
  python tools/test_snippets.py --models D:/comp/bundle/models
  python tools/test_snippets.py imbalance/ modeling/nn  hanya slot yang cocok
  python tools/test_snippets.py --recipes               uji resep per task
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("TEST_GPU", "0"))


def needs_of(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#\s*needs:\s*(.*)", line)
        if m:
            return [x.strip() for x in m.group(1).split(",") if x.strip()]
        if line and not line.startswith("#"):
            break
    return []


def load_chain(slot, seen=None):
    seen = seen if seen is not None else []
    if slot in seen:
        return seen
    path = ROOT / "snippets" / (slot + ".py")
    for n in needs_of(path):
        load_chain(n, seen)
    seen.append(slot)
    return seen


def fixtures(models_dir):
    import numpy as np
    import pandas as pd

    d = ROOT / "datasets"
    tr = pd.read_csv(d / "indonlu_sentiment/train.tsv", sep="\t", header=None, names=["text", "label"]).sample(600, random_state=1).reset_index(drop=True)
    te = pd.read_csv(d / "indonlu_sentiment/test.tsv", sep="\t", header=None, names=["text", "label"]).head(80)
    y = pd.factorize(tr.label, sort=True)[0]
    hate = pd.read_csv(d / "id_multilabel_hate/re_dataset.csv", encoding="latin-1").sample(300, random_state=1)
    nli = pd.read_json(d / "indonli/val.jsonl", lines=True).head(150)
    casa = pd.read_csv(d / "casa/train.csv")
    sents, tags, s, t = [], [], [], []
    for line in open(d / "nergrit/train.txt", encoding="utf-8"):
        line = line.strip()
        if not line:
            if s:
                sents.append(s); tags.append(t); s, t = [], []
        else:
            a = line.split("\t"); s.append(a[0]); t.append(a[-1])
    lang = pd.read_parquet(sorted((d / "lang_id").glob("validation*"))[0]).head(600)
    fx = dict(np=np, pd=pd, texts=tr.text.tolist(), y=y, texts_test=te.text.tolist(), ml_texts=hate.Tweet.tolist(), Y=hate[["HS", "Abusive", "HS_Individual", "HS_Group"]].to_numpy(),
              pa=nli.premise.tolist(), pb=nli.hypothesis.tolist(), py=pd.factorize(nli.label, sort=True)[0], _pd_absa=casa, ner_s=sents, ner_t=tags, lang_t=lang.text.tolist(), lang_y=lang.labels.to_numpy(), MODELS={})
    if models_dir:
        names = {"e5": "e5-small", "indobertweet": "indobertweet", "sentiment": "sent-w11wo", "nli": "indonli-lite", "qa": "indobert-qa", "ner": "nusabert-ner", "t5": "t5-small-sum"}
        fx["MODELS"] = {k: str(Path(models_dir) / v) for k, v in names.items() if (Path(models_dir) / v).exists()}
    try:
        import torch
        fx["DEV"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        fx["DEV"] = "cpu"
    return fx


def run_slot(slot, fx, work):
    ns = {"__name__": "snippet"}
    for s in load_chain(slot):
        exec(compile((ROOT / "snippets" / (s + ".py")).read_text(encoding="utf-8"), s, "exec"), ns)
    tf = ROOT / "tools" / "t" / (slot.replace("/", "__") + ".py")
    if not tf.exists():
        return "TANPA TES"
    env = dict(ns); env.update(fx)
    os.chdir(work)
    exec(compile(tf.read_text(encoding="utf-8"), str(tf), "exec"), env)
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--models")
    ap.add_argument("--recipes", action="store_true")
    a = ap.parse_args()
    slots = sorted(p.relative_to(ROOT / "snippets").with_suffix("").as_posix() for p in (ROOT / "snippets").rglob("*.py"))
    if a.only:
        slots = [s for s in slots if any(o in s for o in a.only)]
    fx = fixtures(a.models)
    print("model:", sorted(fx["MODELS"]), "| device:", fx["DEV"], flush=True)
    work = tempfile.mkdtemp(prefix="snip_")
    res, t0 = {}, time.time()
    for s in slots:
        t = time.time()
        try:
            res[s] = run_slot(s, fx, work)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            tb = traceback.format_exc().strip().splitlines()
            res[s] = "GAGAL: " + " | ".join(tb[-3:])[:400]
        print(f"{res[s][:420]:<8} {time.time() - t:6.1f}s  {s}", flush=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    bad = {k: v for k, v in res.items() if v != "OK"}
    print(f"\n{len(res)} slot, {len(res) - len(bad)} lolos, {len(bad)} bermasalah, {time.time() - t0:.0f} detik")
    (ROOT / "references" / "snippet_test_result.json").write_text(json.dumps({"device": fx["DEV"], "models": sorted(fx["MODELS"]), "python": sys.version.split()[0], "result": res}, indent=1), encoding="utf-8")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
