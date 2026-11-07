# %% [markdown]
# # 24 Summarization
#
# Meringkas teks panjang. Contoh: artikel berita IndoSum, dengan metode ekstraktif dan abstraktif.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta ringkasan, dan metriknya biasanya ROUGE. Untuk merangkum ribuan komentar MBG, jalankan
# per topik: pakai folder `01` dulu, lalu ringkas tiap topik.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_indosum` dan kolom `sentences`, `summary_text`.
# - `SUMMARY_SENTENCES`, sesuaikan panjang ringkasan dengan jawaban.
# - Tanpa `labels` per kalimat, lewati model ekstraktif supervised.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya metode ekstraktif, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
SUMMARY_SENTENCES = 3

SETTINGS = {
    "cepat": {"n_train": 2000, "n_dev": 150, "n_test": 300, "finetune_rows": 0, "finetune_epochs": 0},
    "penuh": {"n_train": 4000, "n_dev": 300, "n_test": 1000, "finetune_rows": 3000, "finetune_epochs": 2},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include indosum

# %% include seq2seq

# %% [markdown]
# ## 4. EDA
#
# Muat latih dan test. File `train.01` dan `test.01` adalah pembagian resmi IndoSum.

# %%
train = load_indosum("train.01.jsonl", SETTINGS["n_train"])
test = load_indosum("test.01.jsonl", SETTINGS["n_test"])
dev = load_indosum("dev.01.jsonl", SETTINGS["n_dev"])
print(f"latih={len(train)} | dev={len(dev)} | test={len(test)}")
stats = pd.DataFrame(
    {"kalimat_artikel": train["sentences"].str.len(),
     "kata_artikel": train["article"].str.split().str.len(),
     "kata_ringkasan": train["summary_text"].str.split().str.len(),
     "kalimat_terpilih": train["labels"].map(sum)}
)
stats.describe(percentiles=[0.5, 0.95]).loc[["50%", "95%", "max"]].round(0)

# %% [markdown]
# Posisi kalimat yang masuk ringkasan. Kalau menumpuk di awal, metode ambil kalimat pertama sudah kuat.

# %%
positions = [i for labels in train["labels"] for i, flag in enumerate(labels) if flag]
fig, ax = plt.subplots()
pd.Series(positions).clip(upper=20).value_counts(normalize=True).sort_index().plot.bar(ax=ax, color="#4C72B0")
ax.set(title="Posisi kalimat yang masuk ringkasan", xlabel="urutan kalimat", ylabel="proporsi")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Temuan dan keputusan**
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 5. Preprocessing
#
# Teks IndoSum sudah bersih, jadi hanya dirapikan. Sumber berita di awal artikel, seperti `Jakarta , CNN Indonesia - -`,
# dibuang karena tidak ada di ringkasan.

# %%
SOURCE_RE = re.compile(r"^[\w ,.()]{0,50}?\s-\s(-\s)?")


def clean_sentence(sentence: str, index: int) -> str:
    return SOURCE_RE.sub("", sentence).strip() if index == 0 else sentence.strip()


def clean_sentences(sentences: list[str]) -> list[str]:
    return [clean_sentence(s, i) for i, s in enumerate(sentences)]


for frame in (train, dev, test):
    frame["sentences"] = frame["sentences"].map(clean_sentences)
    frame["article"] = frame["sentences"].map(" ".join)
print(train["sentences"][0][0][:120])

# %% [markdown]
# ## 6. Validasi
#
# Memakai pembagian resmi IndoSum: dev untuk memilih metode, test hanya untuk laporan. ROUGE ditulis sendiri supaya jalan offline: F1 unigram (ROUGE-1) dan bigram
# (ROUGE-2) setelah huruf kecil dan tanpa tanda baca. Angkanya bisa sedikit berbeda dari paket ROUGE lain.

# %%
from collections import Counter  # noqa: E402

WORD_RE = re.compile(r"\w+")


def rouge_n(prediction: str, reference: str, n: int) -> float:
    def grams(text: str) -> Counter:
        words = WORD_RE.findall(text.lower())
        return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    pred, ref = grams(prediction), grams(reference)
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


assert rouge_n("a b c", "a b c", 1) == 1.0 and rouge_n("a b", "c d", 2) == 0.0
assert abs(rouge_n("a b", "a c", 1) - 0.5) < 1e-9
references = test["summary_text"].tolist()
results: list[dict] = []


def record(name: str, summaries: list[str]) -> None:
    results.append({"metode": name, "rouge1": round(np.mean([rouge_n(p, r, 1) for p, r in zip(summaries, references)]), 4),
                    "rouge2": round(np.mean([rouge_n(p, r, 2) for p, r in zip(summaries, references)]), 4),
                    "kata_rata2": round(np.mean([len(s.split()) for s in summaries]), 1)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Ambil beberapa kalimat pertama, yang sering sulit dikalahkan pada berita. Lalu TextRank tanpa pelatihan, yaitu
# kalimat yang paling mirip dengan kalimat lain dianggap paling penting.

# %%
def lead_summary(sentences: list[str]) -> str:
    return " ".join(sentences[:SUMMARY_SENTENCES])


def textrank_summary(sentences: list[str]) -> str:
    if len(sentences) <= SUMMARY_SENTENCES:
        return " ".join(sentences)
    matrix = TfidfVectorizer().fit_transform(sentences)
    similarity = (matrix @ matrix.T).toarray()
    np.fill_diagonal(similarity, 0)
    transition = similarity / np.maximum(similarity.sum(1, keepdims=True), 1e-9)
    rank = np.ones(len(sentences)) / len(sentences)
    for _ in range(30):
        rank = 0.15 / len(sentences) + 0.85 * transition.T @ rank
    chosen = sorted(np.argsort(-rank)[:SUMMARY_SENTENCES])
    return " ".join(sentences[i] for i in chosen)


record("kalimat_pertama", [lead_summary(s) for s in test["sentences"]])
with stage("textrank"):
    record("textrank", [textrank_summary(s) for s in test["sentences"]])

# %% [markdown]
# ## 8. Model Utama
#
# **Ekstraktif supervised:** classifier per kalimat dari fitur posisi dan kemiripan, dilatih dari label kalimat.
# **Abstraktif:** mT5 menulis ringkasan dengan kata sendiri. Di mode penuh mT5 disetel lagi pada data ini.

# %%
def sentence_features(sentences: list[str]) -> np.ndarray:
    matrix = TfidfVectorizer().fit_transform(sentences)
    centroid = np.asarray(matrix.mean(0)).ravel()
    similarity = matrix @ centroid
    total = len(sentences)
    return np.column_stack(
        [np.arange(total), np.arange(total) / total, similarity, [len(s.split()) for s in sentences],
         [float(i == 0) for i in range(total)], [float(i == 1) for i in range(total)]]
    )


with stage("ekstraktif"):
    x_rows, y_rows = [], []
    for sentences, labels in zip(train["sentences"], train["labels"]):
        if len(sentences) == len(labels) and len(sentences) > 1:
            x_rows.append(sentence_features(sentences))
            y_rows.extend(labels)
    sentence_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack(x_rows), np.array(y_rows, dtype=int))


def extractive_summary(sentences: list[str]) -> str:
    if len(sentences) <= SUMMARY_SENTENCES:
        return " ".join(sentences)
    scores = sentence_model.predict_proba(sentence_features(sentences))[:, 1]
    return " ".join(sentences[i] for i in sorted(np.argsort(-scores)[:SUMMARY_SENTENCES]))


record("ekstraktif_supervised", [extractive_summary(s) for s in test["sentences"]])

# %% [markdown]
# Ringkasan abstraktif. Untuk `csebuetnlp/mT5_multilingual_XLSum`, teks masukan dipotong sesuai batas model.

# %%
abstractive = None
if USE_PRETRAINED:
    tokenizer, model = load_seq2seq("mt5_sum")
    with stage("mt5_generate"):
        abstractive = generate_texts(model, tokenizer, test["article"].tolist(), max_input=512, max_new=128, batch_size=16)
    record("mt5_pretrained", abstractive)
    if SETTINGS["finetune_rows"]:
        with stage("mt5_finetune"):
            subset = train.head(SETTINGS["finetune_rows"])
            model = finetune_seq2seq(model, tokenizer, subset["article"].tolist(), subset["summary_text"].tolist(),
                                     SETTINGS["finetune_epochs"], SEED, batch_size=4, max_input=384, max_target=128)
            abstractive = generate_texts(model, tokenizer, test["article"].tolist(), max_input=512, max_new=128, batch_size=16)
        record("mt5_setelah_finetune", abstractive)
    with stage("mt5_dev"):
        abstractive_dev = generate_texts(model, tokenizer, dev["article"].tolist(), max_input=512, max_new=128, batch_size=16)
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 9. Post-processing
#
# Pilih metode dengan ROUGE-2 tertinggi di **data dev**, bukan di test. Lalu rapikan: buang kalimat berulang di hasil.

# %%
def drop_repeats(summary: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", summary)
    return " ".join(dict.fromkeys(parts))


METHODS = {"ekstraktif_supervised": extractive_summary, "kalimat_pertama": lead_summary, "textrank": textrank_summary}
dev_outputs = {name: [fn(s) for s in dev["sentences"]] for name, fn in METHODS.items()}
test_outputs = {name: [fn(s) for s in test["sentences"]] for name, fn in METHODS.items()}
if abstractive is not None:
    dev_outputs["mt5"], test_outputs["mt5"] = abstractive_dev, abstractive

dev_scores = {name: round(float(np.mean([rouge_n(p, r, 2) for p, r in zip(out, dev["summary_text"])])), 4)
              for name, out in dev_outputs.items()}
best_name = max(dev_scores, key=dev_scores.get)
print("ROUGE-2 dev:", dev_scores, "-> dipilih:", best_name)
final_summaries = [drop_repeats(s) for s in test_outputs[best_name]]
record(f"final_{best_name}", final_summaries)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Skor per kategori** dan contoh hasil. ROUGE tidak bisa menilai kebenaran fakta, jadi periksa ringkasan abstraktif
# secara manual untuk fakta yang dikarang.

# %%
per_doc = pd.DataFrame({"kategori": test["category"], "rouge2": [rouge_n(p, r, 2) for p, r in zip(final_summaries, references)]})
print(per_doc.groupby("kategori")["rouge2"].agg(["size", "mean"]).round(3))
examples = pd.DataFrame({"asli": [r[:130] for r in references[:3]], "hasil": [p[:130] for p in final_summaries[:3]]})
examples

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": test.index, "ringkasan": final_summaries}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("summarization", len(train), {r["metode"]: {"rouge1": r["rouge1"], "rouge2": r["rouge2"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# ISI_SETELAH_EDA
