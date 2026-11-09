# title: N-gram dengan backoff
# when: Baseline next word. Top-k menurut konteks terpanjang yang ada.
# avoid: Split per kata: bocor. Split per dokumen.
# tags: ngram next_word autocomplete
import re
from collections import Counter, defaultdict

class NGramLM:
    def __init__(self, n=3):
        self.n, self.tab = n, [defaultdict(Counter) for _ in range(n)]
    def fit(self, docs):
        for d in docs:
            w = re.findall(r"\w+", str(d).lower())
            for i in range(len(w)):
                for o in range(self.n):
                    if i - o >= 0:
                        self.tab[o][tuple(w[i - o:i])][w[i]] += 1
        return self
    def topk(self, context, k=5):
        w = tuple(re.findall(r"\w+", str(context).lower()))
        out = []
        for o in range(self.n - 1, -1, -1):
            for x, _ in self.tab[o].get(w[-o:] if o else (), Counter()).most_common(k):
                if x not in out:
                    out.append(x)
            if len(out) >= k:
                break
        return out[:k]
