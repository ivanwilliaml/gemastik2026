# title: Labeling function dan leksikon aspek
# when: Tanpa label, ada domain knowledge. Audit hit dan presisi tiap aturan pada 30 sampel.
# avoid: Regex tanpa batas kata. Satu aturan sebagai kebenaran.
# tags: weak labeling_function leksikon aho_corasick voting gold
import re
import numpy as np
import pandas as pd

def keyword_lf(texts, lexicon):
    """lexicon: {nama: [kata]}. Kembalikan matriks hit (n, n_nama) dengan pencocokan batas kata."""
    names = list(lexicon)
    pats = [re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(lexicon[n], key=len, reverse=True))) + r")\b", re.I) for n in names]
    return pd.DataFrame([[len(p.findall(str(t))) for p in pats] for t in texts], columns=names)

def audit_lf(H, texts, k=30, seed=42):
    """Jumlah hit dan sampel acak untuk dibaca."""
    print(H.gt(0).mean().round(4).to_string())
    return {c: pd.Series(list(texts))[H[c] > 0].sample(min(k, int((H[c] > 0).sum())), random_state=seed).tolist() for c in H}

def vote_lf(H, weights=None, abstain=-1):
    """Voting tertimbang. Baris tanpa hit menjadi abstain. weights dari presisi aturan pada gold."""
    A = H.to_numpy(float) * (np.asarray(weights) if weights is not None else 1.0)
    return np.where(A.sum(1) > 0, A.argmax(1), abstain)

def lf_accuracy(pred, gold, abstain=-1):
    pred, gold = np.asarray(pred), np.asarray(gold)
    c = pred != abstain
    return dict(coverage=round(c.mean(), 4), precision=round(float((pred[c] == gold[c]).mean()) if c.any() else 0, 4))
