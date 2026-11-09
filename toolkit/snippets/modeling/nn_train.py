# title: Pelatih neural teks gaya scikit-learn
# when: Dipakai TextCNN, BiGRU, BiLSTM, Attention. Pretrained dilarang atau diversity pada data besar.
# avoid: Data kecil: hafalan (IF26 0,52).
# tags: pytorch neural vocabulary
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

class PassText:
    """Vectorizer kosong untuk oof_predict: model neural menerima teks mentah."""
    def fit_transform(self, texts, y=None): return list(texts)
    def transform(self, texts): return list(texts)

class NNTextClassifier(BaseEstimator, ClassifierMixin):
    """make_net(vocab, n_out) mengembalikan nn.Module yang menerima tensor token (B, L)."""
    def __init__(self, make_net=None, epochs=3, batch=128, lr=1e-3, max_len=128, min_count=2, device=None, seed=42):
        self.make_net, self.epochs, self.batch, self.lr, self.max_len, self.min_count, self.device, self.seed = make_net, epochs, batch, lr, max_len, min_count, device, seed
    def _ids(self, texts):
        ids = np.zeros((len(texts), self.max_len), dtype="int64")
        for i, t in enumerate(texts):
            w = [self.vocab_.get(x, 1) for x in re.findall(r"\w+", str(t).lower())[:self.max_len]]
            ids[i, :len(w)] = w
        return ids
    def fit(self, texts, y):
        torch.manual_seed(self.seed)
        self.dev_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        c = Counter(w for t in texts for w in re.findall(r"\w+", str(t).lower()))
        self.vocab_ = {w: i + 2 for i, (w, n) in enumerate(c.most_common()) if n >= self.min_count}
        self.classes_, yy = np.unique(y, return_inverse=True)
        X, yy = torch.tensor(self._ids(texts)), torch.tensor(yy)
        self.net_ = self.make_net(len(self.vocab_) + 2, len(self.classes_)).to(self.dev_)
        opt, crit = torch.optim.AdamW(self.net_.parameters(), lr=self.lr), nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            self.net_.train()
            for idx in torch.randperm(len(yy)).split(self.batch):
                opt.zero_grad()
                loss = crit(self.net_(X[idx].to(self.dev_)), yy[idx].to(self.dev_))
                loss.backward(); opt.step()
        return self
    @torch.inference_mode()
    def predict_proba(self, texts):
        self.net_.eval()
        X = torch.tensor(self._ids(texts))
        return np.vstack([self.net_(b.to(self.dev_)).softmax(1).float().cpu().numpy() for b in X.split(512)])
    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(1)]
