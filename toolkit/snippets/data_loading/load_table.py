# title: Baca CSV, TSV, Parquet, JSONL
# when: Menit pertama. Otomatis mencoba beberapa encoding dan menyimpan salinan parquet.
# tags: csv parquet jsonl encoding
from pathlib import Path
import pandas as pd

def read_any(path, cache=True, **kw):
    """df = read_any('train.csv'). Lain kali otomatis membaca train.parquet."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, **kw)
    if cache and p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    if p.suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=p.suffix == ".jsonl", **kw)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, **kw)
    sep = kw.pop("sep", "\t" if p.suffix == ".tsv" else ",")
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(p, sep=sep, encoding=enc, on_bad_lines="warn", **kw)
            break
        except UnicodeDecodeError:
            continue
    if cache:
        df.to_parquet(p.with_suffix(".parquet"), index=False)
    return df
