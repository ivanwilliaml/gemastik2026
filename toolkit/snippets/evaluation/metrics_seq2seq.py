# title: CER, WER, BLEU sederhana
# when: Normalisasi teks dan terjemahan. sacrebleu tidak ada di bundle.
# avoid: Menyamakan BLEU sederhana ini dengan versi panitia: cek rumusnya.
# tags: cer wer bleu terjemahan normalisasi
import math
import re
from collections import Counter
from rapidfuzz.distance import Levenshtein

def cer(preds, refs):
    return round(sum(Levenshtein.distance(str(p), str(r)) for p, r in zip(preds, refs)) / max(sum(len(str(r)) for r in refs), 1), 4)

def wer(preds, refs):
    return round(sum(Levenshtein.distance(str(p).split(), str(r).split()) for p, r in zip(preds, refs)) / max(sum(len(str(r).split()) for r in refs), 1), 4)

def bleu(preds, refs, n=4):
    """Corpus BLEU-4 dengan smoothing add-1 untuk n>1."""
    tok = lambda s: re.findall(r"\w+", str(s).lower())
    lp, c_, r_ = 0.0, 0, 0
    for k in range(1, n + 1):
        m, t = 0, 0
        for p, q in zip(preds, refs):
            a, b = tok(p), tok(q)
            A, B = Counter(tuple(a[i:i + k]) for i in range(len(a) - k + 1)), Counter(tuple(b[i:i + k]) for i in range(len(b) - k + 1))
            m += sum((A & B).values()); t += sum(A.values())
            if k == 1:
                c_ += len(a); r_ += len(b)
        lp += math.log((m + (k > 1)) / max(t + (k > 1), 1)) if m or k > 1 else -99
    return round(min(1, math.exp(1 - r_ / max(c_, 1))) * math.exp(lp / n), 4)
