"""Mesin pelatihan: TF-IDF, embedding, transformer, cross-fit, dan pemilihan pseudo-label."""
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from . import core
from .text import EMOJI
from .validate import METRICS, Preds

TOKEN_PATTERN = rf"(?u)\b\w\w+\b|[{EMOJI}]|[!?]"


@dataclass
class Plan:
    """Ukuran pelatihan, dipilih dari batas waktu dan jumlah label."""

    folds: int
    epochs: int
    student_epochs: int
    rounds: int


def plan_training(minutes: float, n_labeled: int) -> Plan:
    """Makin banyak label, makin sedikit epoch yang dibutuhkan."""
    if minutes < 15:
        folds, epochs, student, rounds = 3, 4, 2, 1
    elif minutes < 45:
        folds, epochs, student, rounds = 3, 6, 3, 1
    else:
        folds, epochs, student, rounds = 5, 10, 4, 2
    epochs = int(np.clip(round(5000 / max(n_labeled, 1)), 2, epochs))
    plan = Plan(folds, epochs, min(student, epochs), rounds)
    print(plan)
    return plan


@dataclass
class TrainConfig:
    """Setelan pelatihan transformer. Sebagian diisi EDA, sebagian diuji di bagian 8."""

    max_len: int = 128
    lr: float = 2e-5
    batch_size: int = 32
    class_weight: bool = True
    smoothing: float = 0.0
    layer_decay: float = 1.0
    head_tail: bool = False


@dataclass
class Settings:
    """Semua keputusan dari EDA. Isinya punya nilai bawaan, jadi sel EDA boleh dicopot."""

    train: TrainConfig = field(default_factory=TrainConfig)
    candidate_steps: list[str] = field(
        default_factory=lambda: ["karakter_ulang", "tawa", "hashtag", "slang", "angka", "tanda_baca", "stopword"]
    )
    transformer_steps: list[str] = field(default_factory=list)
    sparse_steps: list[str] = field(default_factory=list)
    shift_auc: float = 0.5
    target_precision: float = 0.9  # presisi pseudo-label yang dikejar
    copy_duplicates: bool = False
    noise_floor: float = 0.0

    @property
    def has_shift(self) -> bool:
        return self.shift_auc >= 0.7


def available_models(names: list[str], model_dir) -> list[str]:
    """Model yang foldernya benar-benar ada. Yang tidak ada dilaporkan, bukan diunduh."""
    found = [n for n in names if (Path(model_dir) / n / "config.json").exists()]
    missing = sorted(set(names) - set(found))
    if missing:
        print(f"model tidak ketemu di {model_dir}: {missing}")
    if not found:
        print("tidak ada model pretrained, seluruh langkah memakai TF-IDF")
    return found


def load_tokenizer(names: list[str], model_dir):
    """Tokenizer model pertama yang tersedia, atau None."""
    usable = available_models(names, model_dir)
    if not usable:
        return None
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(Path(model_dir) / usable[0])


class SparseLearner:
    """TF-IDF kata dan karakter + model linear. Tidak butuh file dari luar."""

    def __init__(self, kind: str, vocabulary_texts: list[str], n_classes: int):
        self.kind, self.n_classes = kind, n_classes
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=TOKEN_PATTERN).fit(vocabulary_texts)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(vocabulary_texts)

    def _features(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def _model(self):
        if self.kind == "nb":
            return ComplementNB(alpha=0.3)
        if self.kind == "svm":
            return LinearSVC(C=0.5, class_weight="balanced")
        return LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced")

    def _probs(self, model, features) -> np.ndarray:
        if self.kind == "svm":
            scores = model.decision_function(features)
            scores = np.column_stack([-scores, scores]) if scores.ndim == 1 else scores
            probs = softmax(scores, axis=1)
        else:
            probs = model.predict_proba(features)
        output = np.zeros((features.shape[0], self.n_classes), dtype=np.float32)
        output[:, model.classes_] = probs
        return output

    def fit_predict(self, train_texts, train_y, predict_sets, weights=None, track=None) -> list[np.ndarray]:
        model = self._model().fit(self._features(train_texts), train_y, sample_weight=weights)
        empty = np.zeros((0, self.n_classes), dtype=np.float32)
        return [self._probs(model, self._features(texts)) if len(texts) else empty for texts in predict_sets]


def _encode(tokenizer, texts, max_len: int, head_tail: bool) -> list[list[int]]:
    """Tokenisasi tanpa padding. head_tail menyimpan 3/4 awal dan 1/4 akhir teks panjang."""
    specials = tokenizer("")["input_ids"]  # token pembuka dan penutup, jalan di semua versi transformers
    prefix, suffix = (specials[:1], specials[1:]) if len(specials) == 2 else ([], specials)
    room = max_len - len(specials)
    encoded = []
    for ids in tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]:
        if len(ids) > room:
            head = room * 3 // 4 if head_tail else room
            ids = ids[:head] + (ids[-(room - head):] if head_tail else [])
        encoded.append(prefix + ids + suffix)
    return encoded


def _batch(sequences, pad_id: int, device: str):
    """Padding hanya sepanjang baris terpanjang di batch, jadi lebih hemat memori."""
    import torch

    longest = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), longest), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
    return ids.to(device), mask.to(device)


def _param_groups(model, config: TrainConfig) -> list[dict]:
    """Layer dekat input diberi learning rate lebih kecil kalau layer_decay di bawah 1."""
    numbers = [int(m.group(1)) for n, _ in model.named_parameters() if (m := re.search(r"layer\.(\d+)\.", n))]
    depth = max(numbers, default=0) + 1
    groups = []
    for name, parameter in model.named_parameters():
        match = re.search(r"layer\.(\d+)\.", name)
        if match:
            power = depth - int(match.group(1))
        elif "embeddings" in name:
            power = depth + 1
        else:
            power = 0
        decay = 0.0 if name.endswith("bias") or "LayerNorm" in name else 0.01
        groups.append({"params": [parameter], "lr": config.lr * config.layer_decay**power, "weight_decay": decay})
    return groups


def _predict(model, tokenizer, sequences, batch_size: int, use_amp: bool, device: str) -> np.ndarray:
    """Probabilitas per baris. Baris diurutkan menurut panjang supaya cepat, lalu dikembalikan."""
    import torch

    order = np.argsort([len(s) for s in sequences])
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size * 2):
            pick = order[start : start + batch_size * 2]
            ids, mask = _batch([sequences[i] for i in pick], tokenizer.pad_token_id, device)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    model.train()
    output = np.zeros((len(sequences), chunks[0].shape[1]), dtype=np.float32) if chunks else np.zeros((0, 1), np.float32)
    if chunks:
        output[order] = np.concatenate(chunks)
    return output


def train_transformer(model_path, train_texts, train_y, predict_sets, n_classes, epochs, seed,
                      config: TrainConfig, weights=None, track=None) -> list[np.ndarray]:
    """Latih satu model klasifikasi, lalu kembalikan probabilitas tiap set."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    device = core.device()
    core.seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=n_classes, ignore_mismatched_sizes=True
    ).to(device)
    encode = lambda texts: _encode(tokenizer, texts, config.max_len, config.head_tail)  # noqa: E731

    train_sequences = encode(train_texts)
    target = torch.tensor(np.asarray(train_y), dtype=torch.long).to(device)
    row_weight = torch.tensor(np.ones(len(train_y)) if weights is None else np.asarray(weights), dtype=torch.float).to(device)
    class_weight = None
    if config.class_weight:
        counts = np.maximum(np.bincount(train_y, minlength=n_classes), 1)
        class_weight = torch.tensor(len(train_y) / (n_classes * counts), dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=config.smoothing, reduction="none")
    optimizer = torch.optim.AdamW(_param_groups(model, config))
    n_batches = max(1, len(train_sequences) // config.batch_size)
    total_steps = max(1, epochs * n_batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_sequences))
        for step in range(n_batches):
            pick = order[step * config.batch_size : (step + 1) * config.batch_size]
            ids, mask = _batch([train_sequences[i] for i in pick], tokenizer.pad_token_id, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = (loss_fn(logits, target[pick]) * row_weight[pick]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        if track is not None:
            probs = _predict(model, tokenizer, encode(track["texts"]), config.batch_size, use_amp, device)
            track["curve"].append(METRICS["macro_f1"](track["y"], probs.argmax(1)))

    empty = np.zeros((0, n_classes), dtype=np.float32)
    outputs = [_predict(model, tokenizer, encode(texts), config.batch_size, use_amp, device) if len(texts) else empty
               for texts in predict_sets]
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return outputs


_EMBED_CACHE: dict = {}


def embed_texts(model_path, texts: list[str], max_len: int = 128, batch_size: int = 64) -> np.ndarray:
    """Embedding mean pooling dari model tanpa dilatih. Hasilnya disimpan supaya tidak dihitung ulang."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    key = (str(model_path), hash(tuple(texts)), max_len)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    hf_logging.set_verbosity_error()
    device = core.device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    sequences = _encode(tokenizer, texts, max_len, False)
    order = np.argsort([len(s) for s in sequences])
    output = None
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            pick = order[start : start + batch_size]
            ids, mask = _batch([sequences[i] for i in pick], tokenizer.pad_token_id, device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            weights = mask.unsqueeze(-1).float()
            pooled = ((hidden * weights).sum(1) / weights.sum(1).clamp(min=1)).float().cpu().numpy()
            if output is None:
                output = np.zeros((len(sequences), pooled.shape[1]), dtype=np.float32)
            output[pick] = pooled
    del model
    torch.cuda.empty_cache()
    _EMBED_CACHE[key] = output
    return output


class Trainer:
    """Satu pintu untuk melatih model apa pun. Nama 'tfidf' memakai SparseLearner."""

    def __init__(self, model_dir, n_classes: int, settings: Settings, vocabulary: list[str]):
        self.model_dir, self.n_classes, self.settings = Path(model_dir), n_classes, settings
        self.vocabulary = vocabulary

    def learner(self, name: str, epochs: int, seed: int, **overrides):
        """Fungsi fit_predict(train_texts, train_y, predict_sets, weights, track) untuk cross_fit."""
        if name == "tfidf":
            return SparseLearner("lr", self.vocabulary, self.n_classes).fit_predict
        config = replace(self.settings.train, **overrides)

        def fit(train_texts, train_y, predict_sets, weights=None, track=None):
            return train_transformer(self.model_dir / name, train_texts, train_y, predict_sets, self.n_classes,
                                     epochs, seed, config, weights, track)

        return fit


def cross_fit(name, fit, texts, y, folds, test_texts=(), pool=(), extra=None, track=False,
              n_classes=None, check_time=True) -> Preds | None:
    """Latih per fold. Kembalikan OOF dan rata-rata test, atau None kalau waktu tidak cukup.

    extra berisi tambahan latih per fold: {fold: (teks, label, bobot)}, dipakai student.
    """
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)
    n_classes = n_classes or int(y.max()) + 1
    fold_ids = np.unique(folds)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    test = np.zeros((len(test_texts), n_classes), dtype=np.float32)
    pool_probs, curve = {}, None
    for position, fold in enumerate(fold_ids):
        begin = time.time()
        train_mask, val_mask = folds != fold, folds == fold
        train_texts, train_y, weights = texts[train_mask].tolist(), y[train_mask], None
        if extra and fold in extra:
            more_texts, more_y, more_weights = extra[fold]
            train_texts = train_texts + list(more_texts)
            train_y = np.concatenate([train_y, more_y])
            weights = np.concatenate([np.ones(int(train_mask.sum())), more_weights])
        sets = [texts[val_mask].tolist(), list(test_texts), list(pool)]
        record = {"texts": sets[0], "y": y[val_mask], "curve": []} if track and position == 0 else None
        pred_val, pred_test, pred_pool = fit(train_texts, train_y, sets, weights, record)
        oof[val_mask] = pred_val
        if len(test_texts):
            test += pred_test / len(fold_ids)
        pool_probs[fold] = pred_pool
        if record:
            curve = record["curve"]
        remaining = (time.time() - begin) * (len(fold_ids) - 1 - position)
        if check_time and position == 0 and core.CLOCK is not None and not core.CLOCK.fits(remaining):
            print(f"{name}: sisa waktu tidak cukup untuk {len(fold_ids) - 1} fold lagi, dilewati")
            return None
    return Preds(name, oof, test, {"pool": pool_probs, "curve": curve})


def blend(name: str, members: list[Preds]) -> Preds:
    """Rata-rata sama rata. Tidak ada bobot yang difit, jadi tidak bisa overfit ke OOF."""
    pool = {}
    if all(m.extra.get("pool") for m in members):
        pool = {fold: np.mean([m.extra["pool"][fold] for m in members], axis=0) for fold in members[0].extra["pool"]}
    return Preds(name, np.mean([m.oof for m in members], axis=0), np.mean([m.test for m in members], axis=0), {"pool": pool})


def class_thresholds(probs: np.ndarray, y: np.ndarray, target: float, floor: float = 0.6, min_rows: int = 10) -> np.ndarray:
    """Ambang keyakinan per kelas supaya presisi di atas target, diukur dari prediksi OOF.

    Kelas yang tidak pernah mencapai target mendapat ambang 1.01, artinya tidak ada pseudo-label kelas itu.
    """
    pred, confidence = probs.argmax(1), probs.max(1)
    thresholds = np.full(probs.shape[1], 1.01)
    for cls in range(probs.shape[1]):
        members = np.where(pred == cls)[0]
        members = members[np.argsort(-confidence[members])]
        if len(members) < min_rows:
            continue
        precision = np.cumsum(y[members] == cls) / np.arange(1, len(members) + 1)
        passing = np.where(precision[min_rows - 1 :] >= target)[0]
        if len(passing):
            thresholds[cls] = max(floor, confidence[members[passing[-1] + min_rows - 1]])
    return thresholds


def select_pseudo(probs: np.ndarray, thresholds: np.ndarray, priors: np.ndarray, agree=None,
                  top_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baris lolos ambang dan disepakati model, dibatasi kuota per kelas sebanding proporsinya."""
    pred, confidence = probs.argmax(1), probs.max(1)
    budget = top_fraction * len(probs)
    picked = []
    for cls, prior in enumerate(priors):
        passed = (pred == cls) & (confidence >= thresholds[cls])
        if agree is not None:
            passed &= agree
        members = np.where(passed)[0]
        picked.append(members[np.argsort(-confidence[members])][: int(budget * prior)])
    index = np.concatenate(picked) if picked else np.array([], dtype=int)
    return index, pred[index], confidence[index]
