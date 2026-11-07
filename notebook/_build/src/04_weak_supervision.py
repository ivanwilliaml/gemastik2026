# %% [markdown]
# # 04 Tanpa Label: Weak Supervision dan Zero-Shot
#
# Kategori sudah ditentukan, tapi tidak ada satu pun data berlabel. Contoh: sentimen SmSA dengan label
# disembunyikan untuk audit, lalu penandaan aspek pada komentar MBG.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Memberi label tanpa data latih, dengan kata kunci dan zero-shot |
# | **Unit prediksi** | Satu komentar mendapat satu kelas sentimen |
# | **Metrik** | Saat lomba tanpa label: distribusi dan kesepakatan sumber. Audit di sini: macro-F1 |
# | **Masalah terbesar** | Tidak ada label untuk memilih model, dan zero-shot mentah sangat bias: hanya 3% prediksi netral. |
# | **Data contoh** | SmSA untuk audit label manusia, komentar MBG untuk demo aspek |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal menyebut kategorinya, tapi datanya tidak berlabel. Misalnya sentimen atau aspek
# komentar MBG. Kalau kategorinya pun belum diketahui, pakai folder `01`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `SEED_WORDS` dan `HYPOTHESES` di config, sesuaikan dengan kategori dan bahasa data.
# - `ASPECTS` untuk penandaan aspek.
# - Sel audit di bagian 10 dihapus, dan ganti dengan cek manual sekitar 100 baris yang diberi label sendiri.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya kata kunci + TF-IDF, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
CLASSES = ["negative", "neutral", "positive"]
HYPOTHESES = {
    "negative": "Teks ini mengungkapkan perasaan negatif, kecewa, atau keluhan.",
    "neutral": "Teks ini hanya menyampaikan informasi atau fakta secara netral.",
    "positive": "Teks ini mengungkapkan perasaan positif, senang, atau pujian.",
}
SEED_WORDS = {
    "negative": "buruk jelek kecewa mengecewakan parah kotor mahal lambat kasar rusak marah benci gagal "
                "tidak enak payah menyesal korupsi bohong".split(),
    "neutral": "akan pada berdasarkan menurut sebesar pukul tanggal bahwa dilakukan digelar diketahui "
               "mengatakan terletak".split(),
    "positive": "bagus enak mantap puas suka nyaman ramah keren senang terima kasih hebat recommended "
                "lezat bersih baik sukses".split(),
}
ASPECTS = {
    "keracunan": "Teks ini membahas keracunan makanan atau siswa yang sakit.",
    "korupsi": "Teks ini membahas korupsi atau penyalahgunaan dana.",
    "anggaran": "Teks ini membahas anggaran, biaya, atau beban keuangan negara.",
    "gizi": "Teks ini membahas gizi, kualitas, atau menu makanan.",
    "dapur": "Teks ini membahas dapur, SPPG, atau pekerja dapur.",
}
PSEUDO_MIN_CONFIDENCE = 0.6
TOP_PER_CLASS_SHARE = 0.25

SETTINGS = {
    "cepat": {"epochs": 2, "model_runs": [("indobertweet", SEED)], "aspect_rows": 500},
    "penuh": {"epochs": 3, "model_runs": [("indobertweet", SEED), ("indobert", SEED)], "aspect_rows": 3000},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include transformer_seqcls

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat data. Label disisihkan dan hanya dipakai untuk audit di bagian akhir.

# %%
import html  # noqa: E402

from sklearn.metrics import f1_score  # noqa: E402


def read_smsa(split: str) -> pd.DataFrame:
    path = EXTRA_DATA_DIR / "indonlu/dataset/smsa_doc-sentiment-prosa" / f"{split}_preprocess.tsv"
    return pd.read_csv(path, sep="\t", header=None, names=["text", "label"])


data = read_smsa("train").sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)
audit_labels = data.pop("label")
class_to_id = {name: i for i, name in enumerate(CLASSES)}
y_audit = audit_labels.map(class_to_id).to_numpy()
print(f"baris: {len(data)}")
text_profile(data["text"])

# %% [markdown]
# Seberapa banyak baris yang tersentuh kata kunci. Kalau kecil, kata kuncinya kurang cocok dengan data.

# %%
def keyword_scores(texts: list[str]) -> np.ndarray:
    """Hitung kecocokan kata kunci per kelas."""
    patterns = [re.compile(r"\b(" + "|".join(SEED_WORDS[c]) + r")\b") for c in CLASSES]
    return np.array([[len(p.findall(t)) for p in patterns] for t in texts], dtype=float)


data["clean"] = data["text"].map(clean_text)
matches = keyword_scores(data["clean"].tolist())
coverage = (matches.sum(1) > 0).mean()
print(f"baris tersentuh kata kunci: {coverage:.1%}")
pd.Series(matches.sum(0), index=CLASSES, name="total_kecocokan")

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Kata kunci hanya menyentuh 71,5% baris | Kata kunci jadi benih, bukan pemutus akhir |
# | Zero-shot mentah memprediksi netral 3% | Skor distandardisasi per kelas, netral naik ke 32% |
# | Sumber lemah sering tidak sepakat | Latih classifier dari 1.127 pseudo-label yang paling yakin |

# %% [markdown]
# ## 5. Preprocessing
#
# Teks hanya dibersihkan ringan, karena model NLI butuh kalimat utuh. Kata kunci sudah memakai huruf kecil.
# Grup near-duplicate tidak dibutuhkan di sini karena tidak ada fold.

# %%
texts = data["clean"].tolist()
print("contoh:", texts[0][:120])

# %% [markdown]
# ## 6. Validasi
#
# Tanpa label, tidak ada skor yang bisa dipakai untuk memilih model saat lomba. Yang bisa dilakukan:
# - Cek distribusi prediksi masuk akal atau tidak.
# - Cek kesepakatan antar sumber lemah. Baris yang disepakati biasanya lebih benar.
# - Label audit di bawah **hanya** untuk laporan dan tidak ikut memilih apa pun.

# %%
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


assert abs(macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) - (2 / 3 + 0.8) / 2) < 1e-9
results: list[dict] = []


def record(name: str, pred: np.ndarray) -> None:
    share = np.bincount(pred, minlength=len(CLASSES)) / len(pred)
    results.append({"model": name, "macro_f1_audit": round(macro_f1(y_audit, pred), 4),
                    "proporsi_prediksi": dict(zip(CLASSES, share.round(2).tolist()))})
    print(results[-1])

# %% [markdown]
# ## 7. Baseline
#
# Kata kunci saja. Baris tanpa kecocokan diberi kelas netral.

# %%
keyword_pred = matches.argmax(1)
keyword_pred[matches.sum(1) == 0] = class_to_id["neutral"]
record("kata_kunci", keyword_pred)

# %% [markdown]
# ## 8. Model Utama
#
# **Zero-shot.** Model NLI menilai seberapa teks mendukung kalimat hipotesis tiap kelas.

# %%
from transformers import AutoModelForSequenceClassification as NliModel  # noqa: E402
from transformers import AutoTokenizer as NliTokenizer  # noqa: E402


@torch.inference_mode()
def zero_shot(texts_in: list[str], hypotheses: list[str], batch_size: int = 64) -> np.ndarray:
    """Probabilitas entailment tiap teks untuk tiap hipotesis, dinormalisasi antar hipotesis."""
    tokenizer = NliTokenizer.from_pretrained(MODEL_DIR / "mdeberta_nli")
    model = NliModel.from_pretrained(MODEL_DIR / "mdeberta_nli").to(DEVICE).eval()
    entail = model.config.label2id.get("entailment", 0)
    scores = np.zeros((len(texts_in), len(hypotheses)))
    for h_index, hypothesis in enumerate(hypotheses):
        for start in range(0, len(texts_in), batch_size):
            batch = texts_in[start : start + batch_size]
            tokens = tokenizer(batch, [hypothesis] * len(batch), padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                logits = model(**tokens).logits.float()
            scores[start : start + batch_size, h_index] = logits[:, entail].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return torch.softmax(torch.tensor(scores), dim=1).numpy()


def standardize_scores(probs: np.ndarray) -> np.ndarray:
    """Standarkan log-probabilitas tiap kategori di seluruh data, lalu kembalikan ke probabilitas.

    Tiap kalimat hipotesis punya bias sendiri, misalnya hipotesis netral hampir selalu dapat skor rendah.
    Standardisasi membuang bias itu tanpa butuh label sama sekali.
    """
    logs = np.log(probs + 1e-9)
    z_scores = (logs - logs.mean(0)) / (logs.std(0) + 1e-9)
    return torch.softmax(torch.tensor(z_scores), dim=1).numpy()


if USE_PRETRAINED:
    with stage("zero_shot"):
        raw_zero = zero_shot(texts, [HYPOTHESES[c] for c in CLASSES])
    record("zero_shot_mentah", raw_zero.argmax(1))
    zero_probs = standardize_scores(raw_zero)
else:
    smoothed = matches + 0.1
    zero_probs = smoothed / smoothed.sum(1, keepdims=True)
record("zero_shot", zero_probs.argmax(1))

# %% [markdown]
# **Weak supervision.** Gabungkan zero-shot dan kata kunci, ambil baris paling yakin di tiap kelas sebagai
# pseudo-label, lalu latih classifier. Hasilnya bisa lebih baik dari sumbernya, karena classifier belajar
# pola umum dari baris yang yakin.

# %%
from sklearn.linear_model import LogisticRegression as WeakLR  # noqa: E402


def pick_confident(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ambil baris paling yakin per kelas dengan kuota sama, supaya kelas langka tetap terwakili."""
    pred, confidence = probs.argmax(1), probs.max(1)
    quota = int(TOP_PER_CLASS_SHARE * len(probs) / probs.shape[1])
    picked = []
    for cls in range(probs.shape[1]):
        members = np.where((pred == cls) & (confidence >= PSEUDO_MIN_CONFIDENCE))[0]
        picked.append(members[np.argsort(-confidence[members])][:quota])
    index = np.concatenate(picked)
    return index, pred[index]


keyword_probs = (matches + 0.2) / (matches + 0.2).sum(1, keepdims=True)
combined = zero_probs * keyword_probs
combined = combined / combined.sum(1, keepdims=True)
index, pseudo = pick_confident(combined)
print("pseudo-label:", dict(zip(CLASSES, np.bincount(pseudo, minlength=3).tolist())))

with stage("classifier_lemah"):
    if USE_PRETRAINED:
        weak_probs = np.zeros_like(zero_probs)
        for model_name, seed in SETTINGS["model_runs"]:
            weak_probs += train_predict(
                model_name, [texts[i] for i in index], pseudo, [texts], num_labels=len(CLASSES),
                epochs=SETTINGS["epochs"], seed=seed,
            )[0] / len(SETTINGS["model_runs"])
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        weak_probs = WeakLR(max_iter=2000, class_weight="balanced").fit(matrix[index], pseudo).predict_proba(matrix)
record("weak_supervision", weak_probs.argmax(1))

# %% [markdown]
# ## 9. Post-processing
#
# Rata-rata zero-shot dan classifier lemah tanpa bobot yang difit, karena tidak ada label untuk memfit
# bobot. Baris yang keyakinannya rendah ditandai supaya bisa dicek manual.

# %%
final_probs = (zero_probs + weak_probs) / 2
final_pred = final_probs.argmax(1)
record("rata_rata_final", final_pred)
low_confidence = final_probs.max(1) < 0.5
print(f"baris keyakinan rendah: {low_confidence.mean():.1%}")

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.** Skor audit hanya bisa dihitung di latihan ini.

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Apakah keyakinan dan kesepakatan bisa dipercaya?** Di soal asli, dua kolom ini yang bisa dipakai untuk
# menilai kualitas tanpa label.

# %%
agree = keyword_pred == zero_probs.argmax(1)
audit_correct = final_pred == y_audit
pd.DataFrame(
    {"kelompok": ["kata_kunci_setuju", "kata_kunci_tidak_setuju", "yakin", "tidak_yakin"],
     "bagian_data": [agree.mean(), (~agree).mean(), (~low_confidence).mean(), low_confidence.mean()],
     "akurasi_audit": [audit_correct[agree].mean(), audit_correct[~agree].mean(),
                       audit_correct[~low_confidence].mean(), audit_correct[low_confidence].mean()]}
).round(3)

# %% [markdown]
# **Confusion matrix audit.**

# %%
plot_confusion(y_audit, final_pred, CLASSES, "Confusion matrix audit")

# %% [markdown]
# **Demo MBG: penandaan aspek.** Zero-shot pada komentar MBG, satu teks bisa punya lebih dari satu aspek.
# Komentar ini tidak punya label, jadi yang ditampilkan adalah contoh untuk dinilai sendiri.

# %%
HTML_RE = re.compile(r"<[^>]+>")
comments = pd.read_csv(find_file("comment_data.csv"))["Comment"].dropna().astype(str)
comments = comments.sample(n=SETTINGS["aspect_rows"], random_state=SEED)
comments = comments.map(lambda t: clean_text(HTML_RE.sub(" ", html.unescape(t)))).reset_index(drop=True)
aspect_names = list(ASPECTS)
if USE_PRETRAINED:
    with stage("aspek_mbg"):
        aspect_probs = standardize_scores(zero_shot(comments.tolist(), [ASPECTS[a] for a in aspect_names]))
    aspect_tags = pd.DataFrame(aspect_probs, columns=aspect_names)
    aspect_tags.insert(0, "komentar", comments.str[:90])
    aspect_tags["aspek_utama"] = aspect_tags[aspect_names].idxmax(axis=1)
    aspect_tags.to_csv(RESULT_DIR / "aspek_mbg.csv", index=False)
    print(aspect_tags["aspek_utama"].value_counts(normalize=True).round(3).to_dict())
    display_rows = aspect_tags.sort_values("korupsi", ascending=False)[["komentar", "aspek_utama", "korupsi"]]
else:
    display_rows = pd.DataFrame({"catatan": ["Lewati, zero-shot butuh model pretrained."]})
display_rows.head(5).round(3)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
submission = pd.DataFrame({"id": data.index, "label": [CLASSES[i] for i in final_pred],
                           "keyakinan": final_probs.max(1).round(3)})
submission.to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("weak_supervision", len(data), {r["model"]: r["macro_f1_audit"] for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Macro-F1 audit naik bertahap: kata kunci 0,4512, zero-shot 0,5584, weak supervision 0,6753, rata-rata akhir 0,6782.
# - Standardisasi sedikit menurunkan macro-F1 audit (0,5769 ke 0,5584), tapi distribusi kelasnya jauh lebih masuk akal.
# - Baris tidak yakin (11,9%) akurasinya 47,1%, sedangkan baris yakin 77,2%. Baris tidak yakin bisa dikirim ke tinjauan manual.
# - Aspek MBG dari zero-shot: korupsi 26,8%, anggaran 24,1%, keracunan 19,3%, dapur 14,9%, gizi 14,9%. Bagian ini tanpa audit.
