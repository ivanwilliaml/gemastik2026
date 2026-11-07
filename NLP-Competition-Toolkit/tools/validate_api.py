"""Jalankan semua panggilan API di katalog metode terhadap Python aktif.

Pakai (dari venv atau PC lomba, setelah bundle terpasang):
  node tools/export_methods.js > methods.json
  python tools/validate_api.py methods.json

Setiap `imp` + `api` benar-benar dieksekusi di folder sementara dengan file contoh kecil.
Nama callable tanpa tanda kurung cukup diperiksa keberadaannya.
Keluaran: daftar panggilan yang gagal, dan kode keluar 1 bila ada yang gagal.
"""
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def fixtures():
    with open("train.csv", "w", encoding="utf-8") as f:
        f.write("text,label\nsatu,0\ndua,1\n")
    with open("train.jsonl", "w", encoding="utf-8") as f:
        f.write('{"text":"satu"}\n{"text":"dua"}\n')
    import pandas as pd

    pd.read_csv("train.csv").to_parquet("train.parquet")


def main():
    methods = json.load(open(sys.argv[1], encoding="utf-8"))
    cwd = tempfile.mkdtemp(prefix="validate_api_")
    os.chdir(cwd)
    fixtures()
    fail, n = [], 0
    for m in methods:
        if not m.get("api"):
            continue
        n += 1
        ns = {}
        try:
            if m.get("imp"):
                exec(m["imp"], ns)
            eval(compile(m["api"], "<api>", "eval"), ns)
        except Exception as e:  # noqa: BLE001 - kita ingin melaporkan semua jenis kegagalan
            # Argumen diterima, tetapi mesin ini tanpa GPU yang mendukung presisi tersebut.
            if "doesn't support bf16" in str(e) or "tf32" in str(e).lower():
                continue
            fail.append((m["id"], m["api"][:90], f"{type(e).__name__}: {str(e)[:170]}"))
    print(f"{n} panggilan diuji, {len(fail)} gagal")
    for f in fail:
        print(" -", *f, sep="\n     ")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
