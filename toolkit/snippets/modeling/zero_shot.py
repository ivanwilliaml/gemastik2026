# title: Zero-shot NLI pada sampel
# when: Label lemah pada sampel, lalu latih classifier. Hipotesis dalam bahasa Indonesia.
# avoid: Semua 500 ribu baris. mDeBERTa belum punya skor Indonesia.
# tags: zero_shot nli mdeberta label_lemah
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@torch.inference_mode()
def zero_shot(model_path, texts, labels, template="Teks ini membahas {}.", multi=False, batch=16, max_len=128, device=None):
    """Kembalikan matriks skor (n_teks, n_label). multi=True: sigmoid per label."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    l2i = {k.lower(): v for k, v in m.config.label2id.items()}
    ent, con = next(v for k, v in l2i.items() if k.startswith("entail")), next(v for k, v in l2i.items() if k.startswith("contra"))
    hyp = [template.format(l) for l in labels]
    out = np.zeros((len(texts), len(labels)), dtype="float32")
    prem = [str(t) for t in texts for _ in labels]
    hh = hyp * len(texts)
    ent_scores = []
    for i in range(0, len(prem), batch):
        e = tok(prem[i:i + batch], hh[i:i + batch], padding=True, truncation="only_first", max_length=max_len, return_tensors="pt").to(device)
        lg = m(**e).logits
        ent_scores.append(lg[:, [con, ent]].softmax(1)[:, 1].float().cpu().numpy() if multi else lg[:, ent].float().cpu().numpy())
    S = np.concatenate(ent_scores).reshape(len(texts), len(labels))
    if multi:
        return S
    S = S - S.max(1, keepdims=True)
    return np.exp(S) / np.exp(S).sum(1, keepdims=True)
