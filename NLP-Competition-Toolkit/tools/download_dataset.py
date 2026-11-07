"""Unduh dataset demonstrasi dari datasets/catalog.json ke datasets/<key>/.

Pakai:
  python tools/download_dataset.py --list
  python tools/download_dataset.py indonlu_sentiment sms_spam
  python tools/download_dataset.py ag_news --sample 20000

Hanya pustaka standar, kecuali --sample yang butuh pandas dan pyarrow.
Jalankan sebelum lomba: butuh internet.
"""
import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((ROOT / "datasets/catalog.json").read_text(encoding="utf-8"))["datasets"]


def size_label(b):
    if not b:
        return "?"
    return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"


def fetch(url, target):
    req = urllib.request.Request(url, headers={"User-Agent": "nlp-toolkit-download"})
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(target)


def from_files(c, out):
    for name, url in c["download"]["files"].items():
        print(f"  {name} <- {url}")
        fetch(url, out / name)


def from_hf(c, out, sample):
    """Ambil parquet hasil konversi Hugging Face, sehingga repo berbasis script pun bisa."""
    d = c["download"]
    url = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(d["repo"], safe="")
    if d.get("config"):
        url += "&config=" + urllib.parse.quote(d["config"], safe="")
    req = urllib.request.Request(url, headers={"User-Agent": "nlp-toolkit-download"})
    with urllib.request.urlopen(req, timeout=60) as r:
        files = json.load(r)["parquet_files"]
    if not files:
        raise RuntimeError("tidak ada file parquet")
    for i, f in enumerate(files):
        target = out / f"{f['split']}-{i:03d}.parquet"
        print(f"  {target.name}: {size_label(f.get('size'))}")
        fetch(f["url"], target)
    if sample:
        import pandas as pd

        for split in {f["split"] for f in files}:
            parts = sorted(out.glob(f"{split}-*.parquet"))
            df = pd.concat(map(pd.read_parquet, parts), ignore_index=True)
            if len(df) > sample:
                df = df.sample(sample, random_state=42)
            for p in parts:
                p.unlink()
            df.to_parquet(out / f"{split}.parquet", index=False)
            print(f"  {split}.parquet: {len(df):,} baris")


def main():
    ap = argparse.ArgumentParser(description="Unduh dataset demonstrasi.")
    ap.add_argument("keys", nargs="*", help="key dari katalog")
    ap.add_argument("--list", action="store_true", help="tampilkan katalog")
    ap.add_argument("--sample", type=int, help="ambil paling banyak N baris acak per split (hanya sumber hf)")
    a = ap.parse_args()

    by_key = {c["key"]: c for c in CATALOG}
    if a.list or not a.keys:
        for c in CATALOG:
            mode = c.get("download", {}).get("type", "manual")
            print(f"{c['key']:22} {size_label(c.get('bytes')):>8}  {mode:7} {c['name']}")
        return

    failed = []
    for key in a.keys:
        c = by_key.get(key)
        if c is None:
            print(f"{key}: key tidak dikenal. Lihat --list")
            failed.append(key)
            continue
        d = c.get("download") or {"type": "manual"}
        if d["type"] == "manual":
            print(f"{key}: unduh manual dari {c['url']}. {d.get('note', '')}")
            continue
        out = ROOT / "datasets" / key
        out.mkdir(parents=True, exist_ok=True)
        print(f"{key} -> datasets/{key}/")
        try:
            if d["type"] == "hf":
                from_hf(c, out, a.sample)
            else:
                from_files(c, out)
        except (OSError, KeyError, ValueError, RuntimeError) as e:
            print(f"  GAGAL: {e}")
            failed.append(key)
    print("Gagal: " + ", ".join(failed) if failed else "Selesai tanpa kegagalan.")
    print("Jalankan python tools/build_library.py agar status lokal diperbarui.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
