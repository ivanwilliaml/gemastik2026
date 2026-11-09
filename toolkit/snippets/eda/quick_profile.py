# title: Profil cepat dan baris kosong
# when: Selalu, menit 10-25. Mencakup pseudo-null seperti 'nan', '-', 'N/A'.
# tags: eda null kosong
import pandas as pd

PSEUDO_NULL = {"", "nan", "none", "null", "-", "--", "n/a", "na"}

def profile(df, text_col=None, n=5, seed=42):
    print("shape", df.shape)
    display_df = pd.DataFrame({"dtype": df.dtypes.astype(str), "null": df.isna().mean().round(4), "nunique": df.nunique()})
    print(display_df.to_string())
    if text_col:
        t = df[text_col].fillna("").astype(str).str.strip()
        print("kosong/pseudo-null:", round(t.str.lower().isin(PSEUDO_NULL).mean(), 4))
        for s in df[text_col].sample(min(n, len(df)), random_state=seed):
            print(" -", str(s)[:200])
    return display_df

def clean_text_col(s):
    """Series teks aman: NaN jadi '', pseudo-null jadi ''."""
    t = s.fillna("").astype(str).str.strip()
    return t.mask(t.str.lower().isin(PSEUDO_NULL), "")
