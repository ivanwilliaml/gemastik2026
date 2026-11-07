# title: EM dan F1 token QA
# when: QA. Normalisasi jawaban Indonesia dan jawaban ganda.
# avoid: Normalisasi artikel Inggris.
# tags: qa em f1 normalisasi
import re
from collections import Counter

def norm_answer(s):
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())

def f1_token(pred, gold):
    p, g = norm_answer(pred).split(), norm_answer(gold).split()
    n = sum((Counter(p) & Counter(g)).values())
    if not p or not g:
        return float(p == g)
    if n == 0:
        return 0.0
    a, b = n / len(p), n / len(g)
    return 2 * a * b / (a + b)

def qa_report(preds, golds):
    """golds: list jawaban atau list daftar jawaban. Ambil skor terbaik."""
    em, f1 = [], []
    for p, g in zip(preds, golds):
        g = g if isinstance(g, (list, tuple)) else [g]
        em.append(max(norm_answer(p) == norm_answer(x) for x in g)); f1.append(max(f1_token(p, x) for x in g))
    return dict(em=round(sum(em) / len(em), 4), f1=round(sum(f1) / len(f1), 4))
