"""Uji pipeline: semua sel tiap task dijalankan berurutan dalam satu proses, seperti notebook.

Pakai:
  python tools/test_pipelines.py
  python tools/test_pipelines.py classification --bert --comp-root D:/comp/bundle
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from pipelines_lib import parse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = ROOT / "datasets"
N = int(os.environ.get("RECIPE_ROWS", 800))


def smsa():
    tr = pd.read_csv(D / "indonlu_sentiment/train.tsv", sep="\t", header=None, names=["text", "label"]).sample(N, random_state=1).reset_index(drop=True)
    te = pd.read_csv(D / "indonlu_sentiment/test.tsv", sep="\t", header=None, names=["text", "label"]).head(100)
    return tr, te


def standard(tr, te, extra=None):
    tr.to_csv("train.csv", index=False)
    te.drop(columns=extra or ["label"], errors="ignore").to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(len(te)), "label": "x"}).to_csv("sample_submission.csv", index=False)


def d_classification():
    tr, te = smsa()
    standard(tr, te)


def d_binary():
    tr, te = smsa()
    tr = tr[tr.label != "neutral"]
    standard(tr, te[te.label != "neutral"])


def d_multilabel():
    h = pd.read_csv(D / "id_multilabel_hate/re_dataset.csv", encoding="latin-1").sample(N, random_state=1)
    d = pd.DataFrame({"text": h.Tweet, "l0": h.HS, "l1": h.Abusive, "l2": h.HS_Individual})
    d.head(N - 100).to_csv("train.csv", index=False)
    d.tail(100)[["text"]].to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(100), "l0": 0, "l1": 0, "l2": 0}).to_csv("sample_submission.csv", index=False)


def d_absa():
    c = pd.read_csv(D / "casa/train.csv").rename(columns={"sentence": "text", "fuel": "a0", "machine": "a1", "price": "a2"})
    c.head(600).to_csv("train.csv", index=False)
    c.tail(100)[["text"]].to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(100), "a0": "x", "a1": "x", "a2": "x"}).to_csv("sample_submission.csv", index=False)


def d_regression():
    g = pd.concat(pd.read_parquet(p) for p in sorted((D / "gplay_review").glob("*.parquet"))).sample(N, random_state=1)
    d = pd.DataFrame({"text": g.text, "label": g.stars.astype(int)})
    standard(d.head(N - 100), d.tail(100))


def d_pair():
    n = pd.read_json(D / "indonli/train.jsonl", lines=True).sample(N, random_state=1)
    d = pd.DataFrame({"text_a": n.premise, "text_b": n.hypothesis, "label": n.label})
    standard(d.head(N - 100), d.tail(100))


def d_semsim():
    s = pd.read_parquet(sorted((D / "stsb").glob("train*"))[0]).sample(N, random_state=1)
    d = pd.DataFrame({"text_a": s.sentence1, "text_b": s.sentence2, "label": s.score})
    standard(d.head(N - 100), d.tail(100))


def d_retrieval():
    c = pd.read_parquet(sorted((D / "indonews").glob("train*"))[0]).head(400).reset_index(drop=True)
    pd.DataFrame({"doc_id": c.index, "text": c.text}).to_csv("docs.csv", index=False)
    q = c.head(40)
    pd.DataFrame({"query_id": q.index, "text": q.text.str.split().str[:12].str.join(" "), "relevant_id": q.index}).to_csv("queries.csv", index=False)


def d_docs():
    c = pd.read_parquet(sorted((D / "indonews").glob("train*"))[0]).sample(N, random_state=1).reset_index(drop=True)
    c["time"] = pd.date_range("2025-01-01", periods=len(c), freq="h").astype(str)
    c[["text", "time", "label"]].to_csv("train.csv", index=False)


def d_weak():
    tr, _ = smsa()
    tr["gold"] = -1
    m = tr.label != "neutral"
    tr.loc[m, "gold"] = (tr.label[m] == "positive").astype(int)
    tr[["text", "gold"]].to_csv("train.csv", index=False)


def d_semi():
    tr, te = smsa()
    tr.loc[tr.sample(frac=.6, random_state=1).index, "label"] = None
    standard(tr, te)


def d_ner():
    for a, b in (("train", "train"), ("valid", "valid"), ("test", "test")):
        Path(b + ".txt").write_text((D / "nergrit" / (a + ".txt")).read_text(encoding="utf-8"), encoding="utf-8")


def d_qa():
    f = pd.read_csv(D / "facqa/train.csv").head(150)
    import ast
    rows = []
    for q, p, s in zip(f.question, f.passage, f.seq_label):
        q, p, s = ast.literal_eval(q), ast.literal_eval(p), ast.literal_eval(s)
        rows.append({"question": " ".join(q), "context": " ".join(p), "answer": " ".join(w for w, t in zip(p, s) if t != "O")})
    d = pd.DataFrame(rows)
    d.head(120).to_csv("train.csv", index=False)
    d.tail(30).drop(columns="answer").to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(30), "answer": "x"}).to_csv("sample_submission.csv", index=False)


def d_summ():
    c = pd.read_parquet(sorted((D / "indonews").glob("train*"))[0]).head(100)
    d = pd.DataFrame({"document": c.text, "summary": c.text.str.split(".").str[0]})
    d.head(80).to_csv("train.csv", index=False)
    d.tail(20).drop(columns="summary").to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(20), "summary": "x"}).to_csv("sample_submission.csv", index=False)


def d_seq2seq():
    d = pd.DataFrame({"source": ["yg bgt gk", "tp udh dpt"] * 40, "target": ["yang banget tidak", "tapi sudah dapat"] * 40})
    d.head(60).to_csv("train.csv", index=False)
    d.tail(20).drop(columns="target").to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(20), "target": "x"}).to_csv("sample_submission.csv", index=False)


def _mbg():
    """Komentar TikTok MBG asli, kotor: kosong, [Sticker], duplikat, label 'empty'. Butuh bundle."""
    root = Path(os.environ.get("COMP_ROOT", "D:/comp/bundle"))
    t = pd.read_csv(root / "data_ext/mbg/MBG_PROCESSED.csv", low_memory=False)
    return t.sample(min(N * 8, len(t)), random_state=1).reset_index(drop=True)


def d_mbg_cls():
    t = _mbg()
    d = pd.DataFrame({"text": t.comment, "label": t.final_sentiment_voting, "time": t.create_time})
    k = len(d) - 800
    d.head(k).to_csv("train.csv", index=False)
    d.tail(800).drop(columns="label").to_csv("test.csv", index=False)
    pd.DataFrame({"id": range(800), "label": "x"}).to_csv("sample_submission.csv", index=False)


def d_mbg_extract():
    t = _mbg()
    pd.DataFrame({"text": t.comment, "time": t.create_time, "sent": t.final_sentiment_voting}).to_csv("train.csv", index=False)


DATA = {"classification": d_classification, "classification_binary": d_binary, "multilabel": d_multilabel, "absa": d_absa, "regression": d_regression, "pair": d_pair, "semsim": d_semsim,
        "retrieval": d_retrieval, "topic": d_docs, "cluster": d_docs, "keyword": d_docs, "report": d_docs, "weak": d_weak, "semi": d_semi, "ner": d_ner, "qa": d_qa,
        "summarization": d_summ, "seq2seq_task": d_seq2seq, "nextword": d_docs, "anomaly": d_docs, "extract": d_mbg_extract, "classification_mbg": d_mbg_cls}


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--bert", action="store_true")
    ap.add_argument("--comp-root")
    a = ap.parse_args()
    fails, report = 0, {}
    for key, prep in DATA.items():
        name = "classification" if key.startswith("classification") else key
        if a.only and not any(o in key for o in a.only):
            continue
        p = parse(ROOT / "pipelines" / (name + ".py"))
        code = "\n\n".join(f"# %% {s['id']}\n" + s["code"] for s in p["stages"])
        if key == "classification_mbg":
            code = code.replace('TEXT, TARGET, TIME = "text", "label", None', 'TEXT, TARGET, TIME = "text", "label", "time"').replace("CHECK_LABEL = False", "CHECK_LABEL = True" if a.comp_root else "CHECK_LABEL = False")
        if key == "extract":
            code = code.replace('"time", None   #', '"time", "sent"   #')
        if a.bert:
            if "USE_BERT = False" not in code and "USE_T5, T5_MODEL = False" not in code:
                continue
            code = code.replace("USE_BERT = False", "USE_BERT = True").replace("USE_T5, T5_MODEL = False", "USE_T5, T5_MODEL = True").replace("BATCH = MODELS / \"indobertweet\", 50000, 128, 32", "BATCH = MODELS / \"indobertweet\", 200, 48, 8")
            code = re.sub(r"(BERT_MODEL, MAX_LEN, BATCH = MODELS / \"[\w-]+\"), \d+, \d+", r"\1, 48, 8", code)
            code = code.replace('MODELS / "indobert-qa", 384, 128', 'MODELS / "indobert-qa", 128, 32')
        work = tempfile.mkdtemp(prefix="pl_")
        os.chdir(work)
        prep()
        Path("all.py").write_text(code, encoding="utf-8")
        t = time.time()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", **({"COMP_ROOT": a.comp_root} if a.comp_root else {})}
        r = subprocess.run([sys.executable, "-W", "ignore", "all.py"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1700, env=env)
        ok = r.returncode == 0
        fails += not ok
        lines = r.stdout.strip().splitlines() or [""]
        print(f"{'OK   ' if ok else 'GAGAL'} {time.time() - t:6.1f}s {key:24} sel={len(p['stages']):2} baris={code.count(chr(10)):4}", flush=True)
        print("   " + (" | ".join(x.strip()[:150] for x in lines[-2:]) if ok else r.stderr.strip()[-900:]), flush=True)
        report[key] = {"ok": ok, "detik": round(time.time() - t, 1), "sel": len(p["stages"]), "bert": bool(a.bert)}
    out = ROOT / "references" / ("pipeline_test_bert.json" if a.bert else "pipeline_test.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
