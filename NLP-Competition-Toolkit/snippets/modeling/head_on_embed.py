# title: LR di atas embedding dan skor model siap pakai
# when: Label sedikit atau waktu sempit. Pembanding wajib sebelum SetFit. Sentimen, emosi siap pakai sebagai fitur.
# avoid: AutoModel: membuang head classifier.
# tags: embedding lr siap_pakai sentimen emosi mlp
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class PrecomputedVec(BaseEstimator, TransformerMixin):
    """Vectorizer untuk oof_predict dari embedding tersimpan. Teks dicari lewat kamus teks ke baris."""
    def __init__(self, texts_all=None, E=None):
        self.texts_all, self.E = texts_all, E
    def fit(self, texts, y=None):
        self.ix_ = {t: i for i, t in enumerate(self.texts_all)}
        return self
    def transform(self, texts):
        return np.asarray(self.E)[[self.ix_[t] for t in texts]]

def make_head(C=1.0):
    return LogisticRegression(C=C, max_iter=2000)

def make_mlp(hidden=(256,), seed=42):
    """Dua lapis di atas embedding, sebagai pembanding LR."""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden, early_stopping=True, max_iter=100, random_state=seed)

@torch.inference_mode()
def pretrained_scores(model_path, texts, max_len=128, batch=64, device=None):
    """Kembalikan (probabilitas, nama label) dari classifier siap pakai."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        e = tok([str(t) for t in texts[i:i + batch]], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out.append(m(**e).logits.softmax(1).float().cpu().numpy())
    return np.vstack(out), [m.config.id2label[i] for i in range(m.config.num_labels)]
