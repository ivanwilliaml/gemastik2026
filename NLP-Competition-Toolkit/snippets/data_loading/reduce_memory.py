# title: Hemat memori DataFrame
# when: Kolom numerik atau kategori besar. Teks tidak disentuh.
# tags: ram memori
import pandas as pd

def reduce_memory(df, cat_ratio=0.1):
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[c] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_string_dtype(s) and s.nunique() / max(len(s), 1) < cat_ratio:
            df[c] = s.astype("category")
    return df
