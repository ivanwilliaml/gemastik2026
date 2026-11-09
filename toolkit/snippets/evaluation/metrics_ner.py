# title: Entity-level F1 tanpa seqeval
# when: NER, POS, term aspek, keyphrase. Exact match span dan tipe.
# avoid: Token accuracy.
# tags: ner f1 span entitas exact_match
from collections import defaultdict

def get_spans(tags):
    out, start, kind = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if start is not None and (t == "O" or t.startswith("B-") or t[2:] != kind):
            out.append((kind, start, i)); start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):
            start, kind = i, t[2:]
    return out

def span_f1(gold, pred):
    """Kembalikan dict micro P/R/F1 dan per tipe."""
    tp, np_, ng = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold, pred):
        G, P = set(get_spans(g)), set(get_spans(p))
        for x in P:
            np_[x[0]] += 1; tp[x[0]] += x in G
        for x in G:
            ng[x[0]] += 1
    def prf(a, b, c):
        p_, r_ = a / max(b, 1), a / max(c, 1)
        return dict(p=round(p_, 4), r=round(r_, 4), f1=round(2 * p_ * r_ / max(p_ + r_, 1e-9), 4))
    res = {"micro": prf(sum(tp.values()), sum(np_.values()), sum(ng.values()))}
    res.update({k: prf(tp[k], np_[k], ng[k]) for k in set(np_) | set(ng)})
    return res

def span_errors(gold, pred):
    """Hitung jenis error: tipe salah, batas salah, terlewat, berlebih."""
    e = dict(tipe=0, batas=0, terlewat=0, berlebih=0)
    for g, p in zip(gold, pred):
        G, P = set(get_spans(g)), set(get_spans(p))
        for x in P - G:
            ov = [y for y in G if y[1] < x[2] and x[1] < y[2]]
            if not ov:
                e["berlebih"] += 1
            elif any(y[1:] == x[1:] for y in ov):
                e["tipe"] += 1
            else:
                e["batas"] += 1
        e["terlewat"] += sum(not any(y[1] < x[2] and x[1] < y[2] for y in P) for x in G - P)
    return e
