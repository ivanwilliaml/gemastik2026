# %% [markdown]
# **Seq2seq.** `generate_texts` menghasilkan teks dari model pretrained. `finetune_seq2seq` melatih model
# itu pada data lu sendiri. Bobot akhir tidak disimpan, supaya folder hasil tetap kecil.

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as SeqTokenizer  # noqa: E402


def load_seq2seq(name: str):
    tokenizer = SeqTokenizer.from_pretrained(MODEL_DIR / name)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR / name).to(DEVICE)
    return tokenizer, model


@torch.inference_mode()
def generate_texts(model, tokenizer, texts, max_input=256, max_new=64, beams=4, batch_size=32, prefix=""):
    """Hasilkan teks, urut dari yang terpanjang supaya padding lebih hemat."""
    model.eval()
    order = np.argsort([-len(t) for t in texts])
    outputs = [""] * len(texts)
    for start in range(0, len(texts), batch_size):
        picks = order[start : start + batch_size]
        tokens = tokenizer([prefix + texts[i] for i in picks], return_tensors="pt", padding=True,
                           truncation=True, max_length=max_input).to(DEVICE)
        generated = model.generate(**tokens, max_new_tokens=max_new, num_beams=beams, no_repeat_ngram_size=3)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, text in zip(picks, decoded):
            outputs[i] = text.strip()
    return outputs


def finetune_seq2seq(model, tokenizer, sources, targets, epochs, seed, lr=3e-5, batch_size=16,
                     max_input=256, max_target=96, prefix=""):
    """Latih model seq2seq dengan loss bawaan."""
    seed_everything(seed)
    source_tokens = tokenizer([prefix + s for s in sources], truncation=True, max_length=max_input,
                              padding="max_length", return_tensors="pt")
    target_ids = tokenizer(text_target=list(targets), truncation=True, max_length=max_target,
                           padding="max_length", return_tensors="pt")["input_ids"]
    target_ids[target_ids == tokenizer.pad_token_id] = -100
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    generator = np.random.default_rng(seed)
    torch.cuda.empty_cache()
    model.train()
    for _ in range(epochs):
        order = generator.permutation(len(sources))
        for start in range(0, len(order), batch_size):
            pick = torch.as_tensor(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            # Potong padding per batch: logit mT5 sangat besar karena kosakatanya 250 ribu.
            attention = source_tokens["attention_mask"][pick]
            source_len = int(attention.sum(1).max())
            target_len = int((target_ids[pick] != -100).sum(1).max())
            loss = model(input_ids=source_tokens["input_ids"][pick][:, :source_len].to(DEVICE),
                         attention_mask=attention[:, :source_len].to(DEVICE),
                         labels=target_ids[pick][:, :target_len].to(DEVICE)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
