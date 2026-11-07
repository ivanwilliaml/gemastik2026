# title: Baseline NER: tag terbanyak, fitur token, gazetteer, CRF
# when: Baseline per token dalam menit. Berlaku juga untuk POS, term aspek, dan keyphrase.
# avoid: BIO ilegal: perbaiki dengan repair_bio.
# tags: ner crf gazetteer most_frequent pos bentuk_kata
from collections import Counter, defaultdict

def most_frequent_tag(sents, tags):
    c = defaultdict(Counter)
    for s, t in zip(sents, tags):
        for w, x in zip(s, t):
            c[w.lower()][x] += 1
    m = {w: v.most_common(1)[0][0] for w, v in c.items()}
    default = Counter(x for t in tags for x in t).most_common(1)[0][0]
    return lambda sents2: [[m.get(w.lower(), default) for w in s] for s in sents2]

def token_features(s, i):
    w = s[i]
    return {"w": w.lower(), "suf3": w[-3:].lower(), "pre3": w[:3].lower(), "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(),
            "w-1": s[i - 1].lower() if i else "<S>", "w+1": s[i + 1].lower() if i < len(s) - 1 else "</S>", "t-1": s[i - 1].istitle() if i else False}

def gazetteer_tag(sents, names, label):
    """names: daftar entitas, cocok persis per token, panjang lebih dulu."""
    g = sorted({tuple(n.lower().split()) for n in names}, key=len, reverse=True)
    out = []
    for s in sents:
        low, t, i = [w.lower() for w in s], ["O"] * len(s), 0
        while i < len(s):
            hit = next((n for n in g if tuple(low[i:i + len(n)]) == n), None)
            if hit:
                t[i:i + len(hit)] = ["B-" + label] + ["I-" + label] * (len(hit) - 1)
                i += len(hit)
            else:
                i += 1
        out.append(t)
    return out

def fit_crf(sents, tags, c1=.1, c2=.1, iters=100):
    """Butuh sklearn-crfsuite. Kembalikan fungsi prediksi."""
    import sklearn_crfsuite
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=iters, all_possible_transitions=True)
    crf.fit([[token_features(s, i) for i in range(len(s))] for s in sents], tags)
    return lambda sents2: [list(x) for x in crf.predict([[token_features(s, i) for i in range(len(s))] for s in sents2])]
