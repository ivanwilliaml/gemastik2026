# title: Seq2seq: ringkasan, terjemahan, normalisasi
# when: T5 Indonesia, atau task terjemahan dan normalisasi. Konfirmasi dulu aturannya.
# avoid: Tanpa baseline ekstraktif atau kamus sebagai pembanding.
# tags: t5 seq2seq terjemahan normalisasi abstraktif
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

@torch.inference_mode()
def generate_texts(model, tok, texts, prefix="", max_len=384, max_new=128, beams=4, batch=8):
    dev, out = next(model.parameters()).device, []
    model.eval()
    for i in range(0, len(texts), batch):
        e = tok([prefix + str(t) for t in texts[i:i + batch]], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        g = model.generate(**e, max_new_tokens=max_new, num_beams=beams, no_repeat_ngram_size=3)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out

def load_seq2seq(model_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    return AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True).to(device), tok

def train_seq2seq(model_path, src, tgt, prefix="", max_src=384, max_tgt=128, batch=8, epochs=1, lr=3e-4, out="run_s2s", seed=42):
    model, tok = load_seq2seq(model_path)
    data = [dict(tok(prefix + str(a), truncation=True, max_length=max_src), labels=tok(text_target=str(b), truncation=True, max_length=max_tgt)["input_ids"]) for a, b in zip(src, tgt)]
    args = Seq2SeqTrainingArguments(output_dir=out, per_device_train_batch_size=batch, num_train_epochs=epochs, learning_rate=lr, save_strategy="no",
                                    report_to="none", dataloader_num_workers=0, seed=seed, disable_tqdm=True)
    Seq2SeqTrainer(model=model, args=args, train_dataset=data, processing_class=tok, data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()
    return model, tok
