# title: ROUGE tanpa paket tambahan
# when: Ringkasan. Uji dengan contoh kecil dan cek rumus panitia.
# avoid: Membandingkan versi rumus berbeda.
# tags: rouge ringkasan lcs
import re
from collections import Counter

def _w(s):
    return re.findall(r"\w+", str(s).lower())

def rouge_n(pred, ref, n=1):
    g = lambda x: Counter(tuple(x[i:i + n]) for i in range(len(x) - n + 1))
    a, b = g(_w(pred)), g(_w(ref))
    m = sum((a & b).values())
    p, r = m / max(sum(a.values()), 1), m / max(sum(b.values()), 1)
    return 2 * p * r / max(p + r, 1e-9)

def rouge_l(pred, ref):
    a, b = _w(pred), _w(ref)
    d = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = d[j + 1]
            d[j + 1] = prev + 1 if x == y else max(d[j + 1], d[j])
            prev = cur
    p, r = d[-1] / max(len(a), 1), d[-1] / max(len(b), 1)
    return 2 * p * r / max(p + r, 1e-9)

def rouge_report(preds, refs):
    n = len(preds)
    return {"rouge1": round(sum(rouge_n(p, r) for p, r in zip(preds, refs)) / n, 4), "rouge2": round(sum(rouge_n(p, r, 2) for p, r in zip(preds, refs)) / n, 4), "rougeL": round(sum(rouge_l(p, r) for p, r in zip(preds, refs)) / n, 4)}
