# title: Tebak kolom id, teks, dan target
# when: Menit pertama untuk mengisi kartu masalah. Tetap verifikasi dengan membaca soal.
# tags: kolom schema
import pandas as pd

def detect_columns(df, test=None):
    rows = []
    for c in df.columns:
        s = df[c]
        is_text = pd.api.types.is_string_dtype(s)
        rows.append({"col": c, "dtype": str(s.dtype), "nunique": s.nunique(), "null": round(s.isna().mean(), 3),
                     "mean_len": round(s.dropna().astype(str).str.len().mean(), 1) if is_text else None,
                     "uniq_rasio": round(s.nunique() / max(len(s), 1), 3),
                     "di_test": None if test is None else c in test.columns})
    out = pd.DataFrame(rows)
    out["tebakan"] = "fitur"
    out.loc[(out.uniq_rasio > .98) & (out.mean_len.fillna(0) < 40), "tebakan"] = "id"
    out.loc[out.mean_len.fillna(0) >= 15, "tebakan"] = "teks"
    if test is not None:
        out.loc[out.di_test == False, "tebakan"] = "target?"  # noqa: E712
    return out.sort_values("tebakan")
