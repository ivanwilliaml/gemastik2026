# title: Normalisasi slang berbatas kata
# when: Saklar satu uji. Hasil lokal -0,0033 sampai +0,0010. Selalu audit jumlah hit.
# avoid: Regex tanpa batas kata: 'git' cocok ke 'gitu'.
# tags: slang alay kamus normalisasi
import re

SLANG_MINI = {"yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "bgt": "banget", "tp": "tapi",
              "klo": "kalau", "udh": "sudah", "dpt": "dapat", "jg": "juga", "jd": "jadi", "bnyk": "banyak", "karna": "karena",
              "krn": "karena", "utk": "untuk", "jgn": "jangan", "org": "orang", "dr": "dari", "lgi": "lagi", "bnr": "benar"}

def load_slang(csv_path=None):
    """csv_path: colloquial-indonesian-lexicon.csv dari bundle. Tanpa file, pakai kamus mini."""
    d = dict(SLANG_MINI)
    if csv_path:
        import pandas as pd
        lex = pd.read_csv(csv_path)
        d.update(dict(zip(lex["slang"].astype(str), lex["formal"].astype(str))))
    return d

def normalize_slang(text, slang=SLANG_MINI, stats=None):
    def rep(m):
        w = m.group(0).lower()
        if w in slang:
            if stats is not None:
                stats[w] = stats.get(w, 0) + 1
            return slang[w]
        return m.group(0)
    return re.sub(r"\b\w+\b", rep, str(text))
