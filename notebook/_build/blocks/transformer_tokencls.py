# %% [markdown]
# **Data format per token.** `read_conll` membaca file satu token per baris dengan label di kolom kedua,
# dan kalimat dipisah baris kosong.

# %%
def read_conll(path: Path) -> list[tuple[list[str], list[str]]]:
    """Kembalikan daftar (token, label) per kalimat."""
    sentences, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if tokens:
                sentences.append((tokens, tags))
            tokens, tags = [], []
            continue
        parts = line.split("\t")
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


# %% [markdown]
# **Metrik tingkat span.** Satu entitas dianggap benar kalau batas dan jenisnya tepat. Ini standar seqeval,
# dan ditulis sendiri supaya tidak butuh library tambahan.

# %%
def extract_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Ubah label BIO menjadi himpunan (mulai, selesai, jenis)."""
    spans, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, name = tag.partition("-")
        if prefix in ("B", "O") or (prefix == "I" and name != kind):
            if start is not None:
                spans.add((start, i, kind))
            start, kind = (i, name) if prefix != "O" else (None, None)
    return spans


def span_scores(true_seqs, pred_seqs) -> dict[str, float]:
    """Precision, recall, dan F1 span secara mikro."""
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(true_seqs, pred_seqs):
        true_spans, pred_spans = extract_spans(true_tags), extract_spans(pred_tags)
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4)}


assert extract_spans(["B-A", "I-A", "O", "B-B"]) == {(0, 2, "A"), (3, 4, "B")}
assert span_scores([["B-A", "I-A", "O"]], [["B-A", "I-A", "O"]])["f1"] == 1.0
assert span_scores([["B-A", "I-A", "O"]], [["B-A", "O", "O"]])["f1"] == 0.0

# %% [markdown]
# **Baseline CRF-lite.** Regresi logistik per token dengan fitur kata, sufiks, bentuk, dan kata tetangga.
# Tidak butuh model dari luar, jadi jadi cadangan kalau pretrained dilarang.

# %%
from sklearn.feature_extraction import DictVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression as TokenLR  # noqa: E402


def token_features(tokens: list[str], i: int) -> dict:
    word = tokens[i]
    return {
        "w": word.lower(), "suf3": word[-3:].lower(), "pre3": word[:3].lower(),
        "title": word.istitle(), "upper": word.isupper(), "digit": word.isdigit(),
        "prev": tokens[i - 1].lower() if i else "<s>",
        "next": tokens[i + 1].lower() if i + 1 < len(tokens) else "</s>",
    }


def token_baseline(train_sents, predict_sents):
    """Latih di token, prediksi per kalimat."""
    x_train = [token_features(s[0], i) for s in train_sents for i in range(len(s[0]))]
    y_train = [t for s in train_sents for t in s[1]]
    vectorizer = DictVectorizer()
    model = TokenLR(max_iter=300, C=3.0)
    model.fit(vectorizer.fit_transform(x_train), y_train)
    output = []
    for tokens, _ in predict_sents:
        features = [token_features(tokens, i) for i in range(len(tokens))]
        output.append(list(model.predict(vectorizer.transform(features))))
    return output


# %% [markdown]
# **Token classification.** Satu kata bisa pecah menjadi beberapa subtoken, jadi label hanya ditaruh di
# subtoken pertama, dan subtoken lain diberi -100 supaya diabaikan oleh loss.

# %%
from transformers import AutoModelForTokenClassification, AutoTokenizer as TokTokenizer  # noqa: E402
from transformers import get_linear_schedule_with_warmup as _token_schedule  # noqa: E402


def encode_tokens(tokenizer, sentences, label_to_id, max_len):
    encoded = tokenizer([s[0] for s in sentences], is_split_into_words=True, truncation=True,
                        max_length=max_len, padding="max_length", return_tensors="pt")
    labels = np.full(encoded["input_ids"].shape, -100)
    for row, (_, tags) in enumerate(sentences):
        previous = None
        for col, word in enumerate(encoded.word_ids(row)):
            if word is not None and word != previous:
                labels[row, col] = label_to_id[tags[word]]
            previous = word
    return encoded, torch.tensor(labels)


@torch.inference_mode()
def predict_tokens(model, tokenizer, sentences, id_to_label, max_len, batch_size=64):
    """Prediksi label per kata. Kata yang terpotong diberi O."""
    model.eval()
    output = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer([s[0] for s in batch], is_split_into_words=True, truncation=True,
                            max_length=max_len, padding=True, return_tensors="pt")
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            logits = model(**{k: v.to(DEVICE) for k, v in encoded.items()}).logits
        pred = logits.argmax(-1).cpu().numpy()
        for row, (tokens, _) in enumerate(batch):
            tags, previous = ["O"] * len(tokens), None
            for col, word in enumerate(encoded.word_ids(row)):
                if word is not None and word != previous:
                    tags[word] = id_to_label[int(pred[row, col])]
                previous = word
            output.append(tags)
    return output


def train_token_model(model_name, train_sents, predict_sets, labels, epochs, seed, max_len=128,
                      lr=3e-5, batch_size=16):
    """Latih model token classification, lalu prediksi tiap set."""
    seed_everything(seed)
    label_to_id = {name: i for i, name in enumerate(labels)}
    id_to_label = {i: name for name, i in label_to_id.items()}
    tokenizer = TokTokenizer.from_pretrained(MODEL_DIR / model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR / model_name, num_labels=len(labels)
    ).to(DEVICE)
    encoded, target = encode_tokens(tokenizer, train_sents, label_to_id, max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(train_sents) // batch_size + 1)
    scheduler = _token_schedule(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    generator = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = generator.permutation(len(train_sents))
        for start in range(0, len(order), batch_size):
            pick = torch.as_tensor(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = model(input_ids=encoded["input_ids"][pick].to(DEVICE),
                             attention_mask=encoded["attention_mask"][pick].to(DEVICE),
                             labels=target[pick].to(DEVICE)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    outputs = [predict_tokens(model, tokenizer, s, id_to_label, max_len) for s in predict_sets]
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return outputs


def repair_bio(tags: list[str]) -> list[str]:
    """Perbaiki urutan tidak sah: I-X tanpa awalan B-X atau I-X di depannya diubah menjadi B-X."""
    fixed, previous = [], "O"
    for tag in tags:
        if tag.startswith("I-") and previous[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        fixed.append(tag)
        previous = tag
    return fixed
