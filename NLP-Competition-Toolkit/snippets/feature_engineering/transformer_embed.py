# title: Mean pooling transformer tanpa fine-tuning
# when: Representasi IndoBERTweet tanpa waktu fine-tune.
# avoid: Lupa mask padding.
# tags: mean_pooling cls embedding transformer
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

@torch.inference_mode()
def mean_pool_embed(model_path, texts, max_len=128, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    m = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), m.config.hidden_size), dtype="float32")
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = m(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        out[idx] = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
    return out
