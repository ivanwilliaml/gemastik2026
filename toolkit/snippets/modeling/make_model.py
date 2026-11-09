# title: make_model(): satu pintu semua model
# when: Ganti nama saja. Klasifikasi: lr, svc, svc_cal, sgd, ridge_clf, cnb, mnb, lgbm, xgb, rf, smote_lr, mlp, dummy, cnn, gru, bigru, lstm, bilstm, attn. Multilabel: ovr, chain. Regresi: ridge, ridge_log, lgbm_reg, lgbm_l1, rf_reg, dummy_reg.
# avoid: Model neural dari nol pada data kecil. cnb pada fitur negatif (SVD, embedding).
# tags: model lr svm naive_bayes lightgbm xgboost random_forest cnn lstm gru attention regresi multilabel smote dummy
import re
from collections import Counter
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def _net(kind, vocab, n_out, dim=200, hidden=128, drop=.3):
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.drop = nn.Embedding(vocab, dim, padding_idx=0), nn.Dropout(drop)
            if kind == "cnn":
                self.conv = nn.ModuleList(nn.Conv1d(dim, hidden, k, padding=k // 2) for k in (3, 4, 5))
                self.out = nn.Linear(hidden * 3, n_out)
            else:
                bi = kind in ("bigru", "bilstm", "attn")
                self.rnn = (nn.LSTM if "lstm" in kind else nn.GRU)(dim, hidden, batch_first=True, bidirectional=bi)
                self.att = nn.Linear(hidden * (2 if bi else 1), 1)
                self.out = nn.Linear(hidden * (2 if bi else 1), n_out)
        def forward(self, x):
            e = self.emb(x)
            if kind == "cnn":
                return self.out(self.drop(torch.cat([c(e.transpose(1, 2)).relu().amax(2) for c in self.conv], 1)))
            h, _ = self.rnn(e)
            w = self.att(h).squeeze(-1).masked_fill(x == 0, -1e4).softmax(1) if kind == "attn" else (x != 0).float() / (x != 0).sum(1, keepdim=True).clamp(min=1)
            return self.out(self.drop((h * w.unsqueeze(-1)).sum(1)))
    return Net()

class NeuralText(BaseEstimator, ClassifierMixin):
    """Terima list teks. Pakai bersama make_vectorizer('raw')."""
    def __init__(self, kind="cnn", epochs=3, batch=128, lr=1e-3, max_len=128, min_count=2, device=None, seed=42):
        self.kind, self.epochs, self.batch, self.lr, self.max_len, self.min_count, self.device, self.seed = kind, epochs, batch, lr, max_len, min_count, device, seed
    def _ids(self, T):
        a = np.zeros((len(T), self.max_len), dtype="int64")
        for i, t in enumerate(T):
            w = [self.v_.get(x, 1) for x in re.findall(r"\w+", str(t).lower())[:self.max_len]]
            a[i, :len(w)] = w
        return a
    def fit(self, T, y):
        import torch
        torch.manual_seed(self.seed)
        self.d_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        c = Counter(w for t in T for w in re.findall(r"\w+", str(t).lower()))
        self.v_ = {w: i + 2 for i, (w, k) in enumerate(c.most_common()) if k >= self.min_count}
        self.classes_, yy = np.unique(y, return_inverse=True)
        X, yy = torch.tensor(self._ids(T)), torch.tensor(yy)
        self.n_ = _net(self.kind, len(self.v_) + 2, len(self.classes_)).to(self.d_)
        opt, loss = torch.optim.AdamW(self.n_.parameters(), lr=self.lr), torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            self.n_.train()
            for b in torch.randperm(len(yy)).split(self.batch):
                opt.zero_grad(); loss(self.n_(X[b].to(self.d_)), yy[b].to(self.d_)).backward(); opt.step()
        return self
    def predict_proba(self, T):
        import torch
        self.n_.eval()
        with torch.inference_mode():
            return np.vstack([self.n_(b.to(self.d_)).softmax(1).float().cpu().numpy() for b in torch.tensor(self._ids(T)).split(512)])
    def predict(self, T):
        return self.classes_[self.predict_proba(T).argmax(1)]

def make_model(name="lr", seed=42, n_jobs=8, **kw):
    if name in ("cnn", "gru", "bigru", "lstm", "bilstm", "attn"):
        return NeuralText(name, seed=seed, **kw)
    from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, SGDClassifier
    if name == "lr": return LogisticRegression(**{**dict(C=4.0, max_iter=2000), **kw})
    if name == "sgd": return SGDClassifier(**{**dict(loss="log_loss", alpha=1e-6, random_state=seed), **kw})
    if name == "ridge_clf": return RidgeClassifier(**kw)
    if name in ("svc", "svc_cal"):
        from sklearn.svm import LinearSVC
        m = LinearSVC(**{**dict(C=.5, dual="auto", max_iter=5000), **kw})
        if name == "svc":
            return m
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(m, method="sigmoid", cv=3)
    if name in ("cnb", "mnb"):
        from sklearn.naive_bayes import ComplementNB, MultinomialNB
        return (ComplementNB if name == "cnb" else MultinomialNB)(**{**dict(alpha=.3), **kw})
    if name in ("lgbm", "lgbm_reg", "lgbm_l1"):
        import lightgbm as lgb
        p = {**dict(n_estimators=500, learning_rate=.05, num_leaves=63, subsample=.8, subsample_freq=1, colsample_bytree=.5, random_state=seed, n_jobs=n_jobs, verbose=-1), **kw}
        return lgb.LGBMClassifier(**p) if name == "lgbm" else lgb.LGBMRegressor(objective="l1" if name == "lgbm_l1" else "l2", **p)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**{**dict(n_estimators=500, learning_rate=.05, max_depth=6, tree_method="hist", subsample=.8, colsample_bytree=.5, random_state=seed), **kw})
    if name in ("rf", "rf_reg"):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return (RandomForestClassifier if name == "rf" else RandomForestRegressor)(**{**dict(n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed), **kw})
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**{**dict(hidden_layer_sizes=(256,), early_stopping=True, max_iter=100, random_state=seed), **kw})
    if name == "smote_lr":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import make_pipeline as imb
        return imb(SMOTE(k_neighbors=kw.get("k", 5), random_state=seed), LogisticRegression(max_iter=2000))
    if name in ("ovr", "chain"):
        base = LogisticRegression(**{**dict(C=4.0, max_iter=2000), **kw})
        if name == "ovr":
            from sklearn.multiclass import OneVsRestClassifier
            return OneVsRestClassifier(base, n_jobs=4)
        from sklearn.multioutput import ClassifierChain
        return ClassifierChain(base, order="random", random_state=seed)
    if name in ("ridge", "ridge_log"):
        m = Ridge(**{**dict(alpha=1.0, solver="sag", random_state=seed), **kw})
        if name == "ridge":
            return m
        from sklearn.compose import TransformedTargetRegressor
        return TransformedTargetRegressor(m, func=np.log1p, inverse_func=np.expm1)
    if name in ("dummy", "dummy_reg"):
        from sklearn.dummy import DummyClassifier, DummyRegressor
        return DummyClassifier(strategy="prior") if name == "dummy" else DummyRegressor(strategy=kw.get("strategy", "mean"))
    raise ValueError(f"model tidak dikenal: {name}")
