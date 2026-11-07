# title: Branch A: pembersihan minimal
# when: Baseline TF-IDF. Tanpa preprocessing juga wajib dicoba.
# avoid: Offset QA atau NER: jangan ubah teks setelah span dihitung.
# tags: lowercase whitespace unicode
import re
import unicodedata

def clean_minimal(text, lower=True, nfkc=True):
    text = str(text)
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lower else text
