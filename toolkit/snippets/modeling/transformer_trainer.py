# title: Fine-tuning transformer: klasifikasi, multilabel, regresi, pasangan
# when: Satu fungsi untuk semua: problem='single', 'multi', 'reg'; texts2 untuk pasangan.
# avoid: 5-fold BERT dan fine-tune 500 ribu baris.
# tags: bert trainer finetune bf16 dynamic_padding early_stopping label_smoothing indobertweet
from pathlib import Path
import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

class TextDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels=None):
        self.enc, self.labels = enc, labels
    def __len__(self):
        return len(self.enc["input_ids"])
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            d["labels"] = self.labels[i]
        return d

class LossTrainer(Trainer):
    """loss_fn(logits, labels) opsional, misalnya focal loss."""
    def __init__(self, *a, loss_fn=None, **k):
        super().__init__(*a, **k)
        self.loss_fn = loss_fn
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self.loss_fn is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = self.loss_fn(out.logits, labels)
        return (loss, out) if return_outputs else loss

def train_transformer(model_path, texts, y, valid_texts=None, valid_y=None, texts2=None, valid_texts2=None, problem="single",
                      num_labels=None, max_len=128, batch=16, accum=1, epochs=1, lr=2e-5, warmup=.1, eval_every=500, patience=3,
                      out="run_tf", seed=42, loss_fn=None, smoothing=0.0, resume=False, fp16=False):
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    def enc(a, b):
        return tok([str(t) for t in a], None if b is None else [str(t) for t in b], truncation=True, max_length=max_len)
    def lab(v):
        v = np.asarray(v)
        return v.astype("float32").tolist() if problem != "single" else v.astype("int64").tolist()
    n = num_labels or (1 if problem == "reg" else np.asarray(y).shape[1] if problem == "multi" else int(np.max(y)) + 1)
    ptype = {"single": "single_label_classification", "multi": "multi_label_classification", "reg": "regression"}[problem]
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True, num_labels=n, problem_type=ptype, ignore_mismatched_sizes=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not fp16
    has_val = valid_texts is not None
    args = TrainingArguments(output_dir=out, per_device_train_batch_size=batch, per_device_eval_batch_size=batch * 2, gradient_accumulation_steps=accum,
                             num_train_epochs=epochs, learning_rate=lr, warmup_steps=warmup, label_smoothing_factor=smoothing,
                             eval_strategy="steps" if has_val else "no", eval_steps=eval_every, save_strategy="steps" if has_val else "no", save_steps=eval_every,
                             save_total_limit=2, load_best_model_at_end=has_val, metric_for_best_model="eval_loss", greater_is_better=False,
                             train_sampling_strategy="group_by_length", bf16=use_bf16, fp16=fp16 and torch.cuda.is_available(),
                             dataloader_num_workers=0, report_to="none", logging_steps=100, seed=seed, disable_tqdm=True)
    tr = LossTrainer(model=model, args=args, train_dataset=TextDS(enc(texts, texts2), lab(y)),
                     eval_dataset=TextDS(enc(valid_texts, valid_texts2), lab(valid_y)) if has_val else None,
                     processing_class=tok, data_collator=DataCollatorWithPadding(tok), loss_fn=loss_fn,
                     callbacks=[EarlyStoppingCallback(patience)] if has_val else None)
    if loss_fn is not None:
        tr.model_accepts_loss_kwargs = False
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(out) if resume and Path(out).is_dir() else None
    if last:
        print("lanjut dari", last)
    tr.train(resume_from_checkpoint=last)
    return tr.model, tok

@torch.inference_mode()
def predict_transformer(model, tok, texts, texts2=None, problem="single", max_len=128, batch=64):
    """Terurut panjang untuk hemat padding. single: softmax, multi: sigmoid, reg: nilai mentah."""
    dev = next(model.parameters()).device
    model.eval()
    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts])
    out = None
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        a, b = [texts[j] for j in idx], None if texts2 is None else [str(texts2[j]) for j in idx]
        e = tok(a, b, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        lg = model(**e).logits
        p = lg.softmax(1) if problem == "single" else lg.sigmoid() if problem == "multi" else lg
        p = p.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), p.shape[1]), dtype="float32")
        out[idx] = p
    return out
