# title: Token classification: latih, prediksi, BIO, voting
# when: NER, POS, term aspek, keyphrase dengan label sendiri, atau NusaBert NER dengan pemetaan label.
# avoid: Label sub-token: pakai -100 kecuali token pertama.
# tags: ner token_classification nusabert word_ids repair_bio voting
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

def _enc(tok, sents, tags, l2i, max_len):
    data = []
    for s, t in zip(sents, tags):
        e = tok(s, is_split_into_words=True, truncation=True, max_length=max_len)
        prev, lab = None, []
        for w in e.word_ids():
            lab.append(-100 if w is None or w == prev else l2i[t[w]])
            prev = w
        e["labels"] = lab
        data.append(dict(e))
    return data

def train_token_model(model_path, sents, tags, batch=16, epochs=3, lr=3e-5, max_len=128, out="run_ner", seed=42):
    labels = sorted({x for t in tags for x in t})
    l2i = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, add_prefix_space=True)
    m = AutoModelForTokenClassification.from_pretrained(str(model_path), local_files_only=True, num_labels=len(labels), id2label={i: l for l, i in l2i.items()}, label2id=l2i, ignore_mismatched_sizes=True)
    bf = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(output_dir=out, per_device_train_batch_size=batch, num_train_epochs=epochs, learning_rate=lr, warmup_steps=.1, save_strategy="no", bf16=bf,
                             dataloader_num_workers=0, report_to="none", seed=seed, disable_tqdm=True)
    tr = Trainer(model=m, args=args, train_dataset=_enc(tok, sents, tags, l2i, max_len), processing_class=tok, data_collator=DataCollatorForTokenClassification(tok))
    tr.train()
    return tr.model, tok

@torch.inference_mode()
def predict_tokens(model, tok, sents, max_len=128, batch=32):
    """Tag per kata asli. Kata yang terpotong mendapat O."""
    dev, id2 = next(model.parameters()).device, model.config.id2label
    model.eval()
    out = []
    for i in range(0, len(sents), batch):
        b = sents[i:i + batch]
        e = tok(b, is_split_into_words=True, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
        p = model(**e).logits.argmax(-1).cpu().numpy()
        for j, s in enumerate(b):
            t, seen = ["O"] * len(s), set()
            for k, w in enumerate(e.word_ids(j)):
                if w is not None and w not in seen:
                    t[w] = id2[int(p[j, k])]; seen.add(w)
            out.append(t)
    return out

def repair_bio(tags):
    """I-X tanpa awalan X menjadi B-X."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t); prev = t
    return out

def vote_tags(models_tags):
    """models_tags: list hasil prediksi tiap model. Voting per token, lalu perbaiki BIO."""
    return [repair_bio([Counter(x).most_common(1)[0][0] for x in zip(*s)]) for s in zip(*models_tags)]
