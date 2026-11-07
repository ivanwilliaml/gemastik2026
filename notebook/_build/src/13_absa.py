# %% [markdown]
# # 13 Aspect-Based Sentiment
#
# Tiap teks punya beberapa aspek, dan tiap aspek punya polaritas sendiri. Contoh: ulasan hotel HoASA,
# tapi polanya sama untuk komentar MBG, misalnya aspek rasa, kebersihan, dan distribusi.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Sentimen per aspek |
# | **Unit prediksi** | Satu ulasan mendapat satu kelas untuk tiap aspek |
# | **Metrik** | Macro-F1 per aspek, lalu dirata-rata |
# | **Masalah terbesar** | Mayoritas label netral karena aspek tidak disebut (59% sampai 92%), dan kelas positif pada banyak aspek di bawah 3%. |
# | **Data contoh** | HoASA: 2.283 latih, 285 valid, 286 test, 10 aspek |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** target berupa polaritas per aspek, misalnya kolom `rasa`, `kebersihan`, `harga` dengan isi
# positif atau negatif. Metriknya biasanya macro-F1 per aspek lalu dirata-rata.
#
# **Yang perlu diganti kalau datanya beda:**
# - `DATA_FOLDER`, `TEXT_COL`, `ASPECTS`, `POLARITIES`.
# - Kalau tiap aspek hanya ada atau tidak ada, pakai folder `12`.
# - Kalau frasa aspeknya harus diambil dari teks, pakai folder `17`.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya TF-IDF per aspek, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
DATA_FOLDER = "hoasa_absa-airy"
TEXT_COL = "review"
ASPECTS = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service", "sunrise_meal", "tv", "wifi"]
POLARITIES = ["neg", "neut", "pos"]

SETTINGS = {
    "cepat": {"epochs": 4, "model_runs": [("indobert", SEED)]},
    "penuh": {"epochs": 8, "model_runs": [("indobert", SEED), ("indobert", SEED + 1), ("indobertweet", SEED)]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat train, valid, dan test. Label langka `neg_pos` berarti satu aspek bercampur, dan digabung ke `neg`
# karena keluhan biasanya lebih penting.

# %%
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402


def read_split(split: str) -> pd.DataFrame:
    path = EXTRA_DATA_DIR / "indonlu/dataset" / DATA_FOLDER / f"{split}_preprocess.csv"
    return pd.read_csv(path).replace({"neg_pos": "neg"}).head(N_ROWS)


train, valid, test = read_split("train"), read_split("valid"), read_split("test")
print(f"latih={len(train)} | valid={len(valid)} | test={len(test)}")
text_profile(train[TEXT_COL])

# %% [markdown]
# Sebaran polaritas per aspek. Aspek yang hampir selalu `neut` adalah sumber masalah utama.

# %%
distribution = pd.DataFrame(
    {a: train[a].value_counts(normalize=True) for a in ASPECTS}
).T.reindex(columns=POLARITIES).fillna(0).round(3)
fig, ax = plt.subplots(figsize=(8, 3.8))
distribution.plot.barh(stacked=True, ax=ax, color=["#C44E52", "#BBBBBB", "#55A868"])
ax.set(title="Polaritas per aspek", xlabel="proporsi", ylabel="aspek")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
distribution

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Netral sangat dominan | Tebak netral sudah menjadi baseline, macro-F1 0,2956 |
# | Aspek `general` paling samar | Laporkan F1 per aspek, jangan hanya rata-rata |
# | Data latih kecil | Satu model berbagi encoder untuk semua aspek |

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan, dan label diubah menjadi matriks angka berukuran baris kali aspek.

# %%
polarity_to_id = {name: i for i, name in enumerate(POLARITIES)}
for frame in (train, valid, test):
    frame["clean"] = frame[TEXT_COL].map(clean_text)
y_train = train[ASPECTS].replace(polarity_to_id).to_numpy(dtype=int)
y_valid = valid[ASPECTS].replace(polarity_to_id).to_numpy(dtype=int)
y_test = test[ASPECTS].replace(polarity_to_id).to_numpy(dtype=int)
print("ukuran label:", y_train.shape)

# %% [markdown]
# ## 6. Validasi
#
# Memakai split valid dan test yang disediakan. Valid untuk memilih, test hanya untuk laporan. Metrik utama
# macro-F1 yang dihitung per aspek lalu dirata-rata, sama seperti evaluasi resmi IndoNLU.

# %%
def aspect_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = [f1_score(y_true[:, a], y_pred[:, a], average="macro", zero_division=0)
              for a in range(y_true.shape[1])]
    return float(np.mean(scores))


_check = np.array([[0], [1]])
assert aspect_macro_f1(_check, _check) == 1.0
results: list[dict] = []


def record(name: str, pred_valid, pred_test) -> None:
    results.append({"model": name, "valid": round(aspect_macro_f1(y_valid, pred_valid), 4),
                    "test": round(aspect_macro_f1(y_test, pred_test), 4)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Tebak `neut` di semua aspek, lalu TF-IDF + regresi logistik untuk tiap aspek.

# %%
neutral = polarity_to_id["neut"]
record("semua_netral", np.full_like(y_valid, neutral), np.full_like(y_test, neutral))

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
x_train = vectorizer.fit_transform(train["clean"])
x_valid, x_test = vectorizer.transform(valid["clean"]), vectorizer.transform(test["clean"])
tfidf_valid, tfidf_test = np.zeros_like(y_valid), np.zeros_like(y_test)
for a in range(len(ASPECTS)):
    model = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced").fit(x_train, y_train[:, a])
    tfidf_valid[:, a], tfidf_test[:, a] = model.predict(x_valid), model.predict(x_test)
record("tfidf_per_aspek", tfidf_valid, tfidf_test)

# %% [markdown]
# ## 8. Model Utama
#
# **Satu encoder dengan banyak kepala.** Encoder dipakai bersama, dan tiap aspek punya lapisan keluaran
# sendiri. Lebih cepat daripada melatih satu model per aspek, dan aspek langka terbantu oleh aspek lain.
# Bobot kelas dipasang per aspek supaya model tidak hanya menebak `neut`.

# %%
class MultiHeadClassifier(torch.nn.Module):
    def __init__(self, model_name: str, n_aspects: int, n_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_DIR / model_name)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, n_aspects * n_classes)
        self.shape = (n_aspects, n_classes)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(hidden[:, 0])).view(-1, *self.shape)


@torch.inference_mode()
def predict_probs(model, ids, mask, batch_size=64) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(ids), batch_size):
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            logits = model(ids[start : start + batch_size].to(DEVICE), mask[start : start + batch_size].to(DEVICE))
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(chunks).numpy()


def train_multihead(model_name: str, epochs: int, seed: int, max_len=128, batch_size=32):
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / model_name)

    def encode(frame):
        tokens = tokenizer(frame["clean"].tolist(), padding="max_length", truncation=True,
                           max_length=max_len, return_tensors="pt")
        return tokens["input_ids"], tokens["attention_mask"]

    ids, mask = encode(train)
    target = torch.tensor(y_train)
    model = MultiHeadClassifier(model_name, len(ASPECTS), len(POLARITIES)).to(DEVICE)
    weights = np.stack([len(y_train) / (len(POLARITIES) * np.maximum(np.bincount(y_train[:, a], minlength=3), 1))
                        for a in range(len(ASPECTS))])
    weights = torch.tensor(np.clip(weights, 0.2, 10), dtype=torch.float).to(DEVICE)
    loader = DataLoader(TensorDataset(ids, mask, target), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    model.train()
    for _ in range(epochs):
        for batch_ids, batch_mask, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_y = batch_y.to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                logits = model(batch_ids.to(DEVICE), batch_mask.to(DEVICE))
                loss = sum(torch.nn.functional.cross_entropy(logits[:, a], batch_y[:, a], weight=weights[a])
                           for a in range(len(ASPECTS))) / len(ASPECTS)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    outputs = [predict_probs(model, *encode(f)) for f in (valid, test)]
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return outputs


probs_valid = np.zeros((*y_valid.shape, len(POLARITIES)))
probs_test = np.zeros((*y_test.shape, len(POLARITIES)))
with stage("transformer"):
    if USE_PRETRAINED:
        for model_name, seed in SETTINGS["model_runs"]:
            run_valid, run_test = train_multihead(model_name, SETTINGS["epochs"], seed)
            probs_valid += run_valid / len(SETTINGS["model_runs"])
            probs_test += run_test / len(SETTINGS["model_runs"])
    else:
        probs_valid = np.eye(len(POLARITIES))[tfidf_valid]
        probs_test = np.eye(len(POLARITIES))[tfidf_test]
record("model_utama", probs_valid.argmax(-1), probs_test.argmax(-1))

# %% [markdown]
# ## 9. Post-processing
#
# Pengali polaritas per aspek, difit di valid. Karena valid kecil, grid dibuat kasar supaya tidak overfit,
# dan hasilnya hanya dipakai kalau valid naik cukup jelas.

# %%
def fit_aspect_multipliers(probs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cari pengali per aspek dan polaritas dengan pencarian bergantian."""
    multipliers = np.ones(probs.shape[1:])
    grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for a in range(probs.shape[1]):
        for cls in range(probs.shape[2]):
            scores = []
            for value in grid:
                trial = multipliers[a].copy()
                trial[cls] = value
                pred = (probs[:, a] * trial).argmax(1)
                scores.append(f1_score(target[:, a], pred, average="macro", zero_division=0))
            multipliers[a, cls] = grid[int(np.argmax(scores))]
    return multipliers


half = len(y_valid) // 2
half_multipliers = fit_aspect_multipliers(probs_valid[:half], y_valid[:half])
honest_gain = aspect_macro_f1(y_valid[half:], (probs_valid[half:] * half_multipliers[None]).argmax(-1)) - \
    aspect_macro_f1(y_valid[half:], probs_valid[half:].argmax(-1))
multipliers = fit_aspect_multipliers(probs_valid, y_valid)
use_multipliers = honest_gain > 0.005
final_valid = (probs_valid * multipliers[None]).argmax(-1) if use_multipliers else probs_valid.argmax(-1)
final_test = (probs_test * multipliers[None]).argmax(-1) if use_multipliers else probs_test.argmax(-1)
record("model_utama_plus_pengali" if use_multipliers else "model_utama_final", final_valid, final_test)
print(f"kenaikan jujur di separuh valid: {honest_gain:+.4f} -> pakai pengali: {use_multipliers}")

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Skor per aspek.** Aspek dengan sedikit data non-netral biasanya paling lemah.

# %%
pd.DataFrame(
    {"non_netral_latih": (y_train != neutral).sum(0),
     "f1_test": [round(f1_score(y_test[:, a], final_test[:, a], average="macro", zero_division=0), 3)
                 for a in range(len(ASPECTS))]},
    index=ASPECTS,
).sort_values("f1_test")

# %% [markdown]
# **Confusion matrix gabungan** semua aspek pada test.

# %%
plot_confusion(y_test.ravel(), final_test.ravel(), POLARITIES, "Confusion matrix gabungan aspek")

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame(np.array(POLARITIES)[final_test], columns=ASPECTS).rename_axis("id").to_csv(RESULT_DIR / "submission.csv")
run_summary("absa", len(train), {r["model"]: {"valid": r["valid"], "test": r["test"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Model utama test 0,8494, jauh di atas TF-IDF per aspek 0,7117.
# - Pengali lolos syarat di separuh valid, tapi test turun sedikit ke 0,8412. Selisih ini kecil, dan valid 0,8778 terlalu optimistis karena pengali juga difit di valid.
# - Aspek terlemah `general` (0,616), terkuat `ac` (0,951).
# - Data ulasan hotel. Untuk MBG, aspek bisa diganti menjadi rasa, kebersihan, gizi, distribusi, dan anggaran.
