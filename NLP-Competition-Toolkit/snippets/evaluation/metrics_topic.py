# title: Coherence NPMI, c_v, dan diversity
# when: Memilih k topik. Selalu baca isinya.
# avoid: Coherence saja. OCTIS.
# tags: coherence npmi diversity topik
import re
import numpy as np

def topic_diversity(topics, top=10):
    w = [x for t in topics for x in t[:top]]
    return round(len(set(w)) / max(len(w), 1), 4)

def npmi_coherence(topics, docs, top=10, eps=1e-12):
    """Kemunculan bersama pada tingkat dokumen. Tanpa paket tambahan."""
    sets = [set(re.findall(r"\w+", str(d).lower())) for d in docs]
    vocab = {w for t in topics for w in t[:top] for w in w.split()}
    inv = {w: {i for i, s in enumerate(sets) if w in s} for w in vocab}
    n = len(sets)
    out = []
    for t in topics:
        ws = [w for w in t[:top] if inv.get(w)]
        vals = []
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = inv[ws[i]], inv[ws[j]]
                p = len(a & b) / n
                vals.append(-1.0 if p == 0 else np.log((p + eps) / (len(a) / n * len(b) / n)) / -np.log(p + eps))
        out.append(float(np.mean(vals)) if vals else 0.0)
    return round(float(np.mean(out)), 4), [round(x, 4) for x in out]

def cv_coherence(topics, docs, top=10):
    """c_v gensim. Lebih lambat."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    toks = [re.findall(r"\w+", str(d).lower()) for d in docs]
    return CoherenceModel(topics=[t[:top] for t in topics], texts=toks, dictionary=Dictionary(toks), coherence="c_v", processes=1).get_coherence()
