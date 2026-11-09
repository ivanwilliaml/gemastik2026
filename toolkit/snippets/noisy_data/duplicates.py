# title: Duplikat, copypasta, dan dedup
# when: Selalu sebelum split. Laporkan volume asli vs copypasta.
# avoid: Drop duplikat di test.
# tags: duplikat copypasta dedup
import re
import pandas as pd

def norm_text(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()

def duplicate_report(texts, k=20):
    s = pd.Series(list(texts)).map(norm_text)
    vc = s.value_counts()
    print("duplikat mentah", round(pd.Series(list(texts)).duplicated().mean(), 4), "| ternormalisasi", round(s.duplicated().mean(), 4))
    return vc.head(k)

def dedup_index(texts):
    """Indeks baris unik pertama, untuk data latih saja."""
    return (~pd.Series(list(texts)).map(norm_text).duplicated()).to_numpy().nonzero()[0]
