# title: P@K, R@K, MAP, MRR, NDCG, kata kunci F1@K
# when: Ranking, retrieval, kata kunci.
# avoid: K tidak sesuai soal.
# tags: precision recall map mrr ndcg retrieval keyword
import numpy as np

def retrieval_report(ranked, relevant, k=10):
    """ranked: list daftar id terurut per query. relevant: list set id (atau dict id->gain untuk NDCG)."""
    P, R, AP, RR, N = [], [], [], [], []
    for r, rel in zip(ranked, relevant):
        gain = rel if isinstance(rel, dict) else {x: 1 for x in rel}
        rs = [x for x in gain if gain[x] > 0]
        hit = [x in rs for x in r]
        P.append(sum(hit[:k]) / k)
        R.append(sum(hit[:k]) / max(len(rs), 1))
        c = np.cumsum(hit)
        AP.append(float(np.sum([c[i] / (i + 1) for i, h in enumerate(hit) if h])) / max(len(rs), 1))
        RR.append(next((1 / (i + 1) for i, h in enumerate(hit) if h), 0))
        dcg = sum((2 ** gain.get(x, 0) - 1) / np.log2(i + 2) for i, x in enumerate(r[:k]))
        idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(sorted(gain.values(), reverse=True)[:k]))
        N.append(dcg / idcg if idcg else 0)
    return {f"p@{k}": round(float(np.mean(P)), 4), f"r@{k}": round(float(np.mean(R)), 4), "map": round(float(np.mean(AP)), 4), "mrr": round(float(np.mean(RR)), 4), f"ndcg@{k}": round(float(np.mean(N)), 4)}

def keyword_f1_at_k(pred, gold, k=5):
    """pred: list daftar kata kunci. gold: list daftar emas. Pencocokan ternormalisasi."""
    tp = np.array([len({x.lower() for x in p[:k]} & {x.lower() for x in g}) for p, g in zip(pred, gold)])
    prec = tp.sum() / max(sum(min(k, len(p)) for p in pred), 1)
    rec = tp.sum() / max(sum(len(g) for g in gold), 1)
    return round(2 * prec * rec / max(prec + rec, 1e-9), 4)
