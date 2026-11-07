# title: Baris tak bermakna, outlier, noise domain
# when: Hanya emoji, [Sticker], rasio non-huruf tinggi, spam judi.
# avoid: Membuang dari test. Aturan tanpa audit pada 30 sampel.
# tags: outlier sticker emoji judi aho_corasick
import re
import numpy as np
import pandas as pd

def text_flags(texts, min_chars=3, max_chars=None):
    s = pd.Series(list(texts)).fillna("").astype(str)
    n = s.str.len().clip(lower=1)
    alpha = s.str.count(r"[^\W\d_]") / n
    d = pd.DataFrame({"kosong": s.str.strip().eq(""), "pendek": n < min_chars, "non_huruf": alpha < .5, "sticker": s.str.contains(r"\[sticker\]", case=False, regex=True),
                      "panjang": n > (max_chars or n.quantile(.999) * 1.0 + 1), "url_saja": s.str.fullmatch(r"\s*https?://\S+\s*")})
    print(d.mean().round(4).to_string())
    return d

def domain_noise(texts, terms):
    """Saring kata seperti judi atau slot dengan Aho-Corasick berbatas kata. Audit hasilnya."""
    import ahocorasick
    A = ahocorasick.Automaton()
    for t in terms:
        A.add_word(f" {t.lower()} ", t)
    A.make_automaton()
    return np.array([any(True for _ in A.iter(" " + " ".join(re.findall(r"\w+", str(x).lower())) + " ")) for x in texts])
