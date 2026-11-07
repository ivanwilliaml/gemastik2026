# %% [markdown]
# # 11 Domain Shift
#
# Data latih dan data target berasal dari domain berbeda. Contoh: latih dari ulasan dan berita (SmSA),
# target komentar YouTube tentang MBG.
#
# **Peringatan data:** label target di sini **hasil keluaran model lain**, bukan anotasi manusia, dan
# hanya dipakai untuk audit. Skor audit perlu dibaca sebagai petunjuk arah, bukan kebenaran.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** AUC adversarial antara train dan test tinggi, misalnya di atas 0,7, atau soal memang
# menyebut sumber data berbeda. Tanda khasnya skor validasi bagus tapi skor leaderboard jatuh.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_source` dan `load_target`.
# - `ALLOW_TARGET_TRAINING`, cek dulu apakah aturan lomba membolehkan data test dipakai saat training.
# - Sel audit di bagian 10 dihapus kalau label target tidak ada.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya TF-IDF, tanpa file model dari luar
ALLOW_TARGET_TRAINING = True  # hanya kalau aturan lomba membolehkan
SEED = 42
N_ROWS = 5000
GUARDRAIL_SHARE = 0.15
AUDIT_MIN_CONFIDENCE = 0.8
CLASSES = ["negative", "neutral", "positive"]
PSEUDO_TOP_FRACTION = 0.4
PSEUDO_MIN_CONFIDENCE = 0.7
MIN_GUARD_GAIN = 0.002  # student dipakai hanya kalau skor penjaga naik minimal sebesar ini

SETTINGS = {
    "cepat": {"epochs": 2, "student_epochs": 2, "rounds": 1, "model_runs": [("indobertweet", SEED)]},
    "penuh": {"epochs": 3, "student_epochs": 2, "rounds": 2,
              "model_runs": [("indobertweet", SEED), ("indobert", SEED), ("indobertweet", SEED + 1)]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include transformer_seqcls

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat sumber dan target. Label target disisihkan untuk audit.

# %%
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def load_source() -> pd.DataFrame:
    path = EXTRA_DATA_DIR / "indonlu/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv"
    frame = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    return frame.sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)


def load_target() -> pd.DataFrame:
    frame = pd.read_csv(find_file("comment_sentiment.csv")).dropna(subset=["Cleaned Comment"])
    frame = frame.rename(columns={"Cleaned Comment": "text", "Sentiment": "label", "Confidence Score": "conf"})
    return frame[["text", "label", "conf"]].head(N_ROWS).reset_index(drop=True)


source, target = load_source(), load_target()
target_labels, target_conf = target.pop("label"), target.pop("conf")
print(f"sumber={len(source)} | target={len(target)}")
pd.DataFrame({"sumber": text_profile(source["text"]), "target": text_profile(target["text"])})

# %% [markdown]
# **Seberapa besar pergeserannya?** AUC adversarial dan kata yang paling membedakan kedua domain.

# %%
shift_auc = adversarial_auc(source["text"].tolist(), target["text"].tolist())
print(f"AUC adversarial sumber vs target: {shift_auc:.3f}")
vectorizer = TfidfVectorizer(min_df=5, token_pattern=r"\b[a-z]{3,}\b")
matrix = vectorizer.fit_transform(pd.concat([source["text"].str.lower(), target["text"].str.lower()]))
domain = np.array([0] * len(source) + [1] * len(target))
coef = LogisticRegression(max_iter=1000).fit(matrix, domain).coef_[0]
vocab = np.array(vectorizer.get_feature_names_out())
print("khas sumber:", ", ".join(vocab[np.argsort(coef)[:12]]))
print("khas target:", ", ".join(vocab[np.argsort(-coef)[:12]]))

# %% [markdown]
# **Temuan dan keputusan**
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan dan **sama persis** di kedua domain. Slang tetap dipertahankan, karena IndoBERTweet
# dilatih dari teks informal seperti target.

# %%
class_to_id = {name: i for i, name in enumerate(CLASSES)}
source["clean"] = source["text"].map(clean_text)
target["clean"] = target["text"].map(clean_text)
target_texts = target["clean"].tolist()
y_target = target_labels.map(class_to_id).to_numpy()
fit_rows, guard_rows = train_test_split(
    source, test_size=GUARDRAIL_SHARE, stratify=source["label"], random_state=SEED
)
fit_texts = fit_rows["clean"].tolist()
guard_texts = guard_rows["clean"].tolist()
y_fit = fit_rows["label"].map(class_to_id).to_numpy()
y_guard = guard_rows["label"].map(class_to_id).to_numpy()
print(f"latih={len(fit_texts)} | penjaga_sumber={len(guard_texts)} | target={len(target_texts)}")

# %% [markdown]
# ## 6. Validasi
#
# Dua alat ukur, dan keduanya bukan cross-validation biasa karena CV di sumber tidak mewakili target:
# - **Skor penjaga di sumber:** memastikan adaptasi tidak merusak kemampuan dasar model.
# - **Distribusi prediksi target:** harus masuk akal, tidak semua masuk satu kelas.
# - **Skor audit target:** hanya untuk laporan. Di soal asli tidak ada.

# %%
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


assert abs(macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) - (2 / 3 + 0.8) / 2) < 1e-9
sure = target_conf.to_numpy() >= AUDIT_MIN_CONFIDENCE
print(f"baris audit yakin: {sure.mean():.1%}")
results: list[dict] = []


def record(name: str, guard_probs: np.ndarray, target_probs: np.ndarray) -> None:
    pred = target_probs.argmax(1)
    share = np.bincount(pred, minlength=len(CLASSES)) / len(pred)
    results.append(
        {"model": name, "penjaga_sumber": round(macro_f1(y_guard, guard_probs.argmax(1)), 4),
         "audit_target": round(macro_f1(y_target, pred), 4),
         "audit_target_yakin": round(macro_f1(y_target[sure], pred[sure]), 4),
         "proporsi_target": dict(zip(CLASSES, share.round(2).tolist()))}
    )
    print(results[-1])

# %% [markdown]
# ## 7. Baseline
#
# TF-IDF dilatih hanya dari sumber. Skor audit di target menunjukkan seberapa parah dampak pergeseran.

# %%
word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
x_fit = word_tfidf.fit_transform(fit_texts)
tfidf_model = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced").fit(x_fit, y_fit)
guard_tfidf = tfidf_model.predict_proba(word_tfidf.transform(guard_texts))
target_tfidf = tfidf_model.predict_proba(word_tfidf.transform(target_texts))
record("tfidf_sumber_saja", guard_tfidf, target_tfidf)

# %% [markdown]
# ## 8. Model Utama
#
# **Self-training ke domain target.** Teacher dilatih dari sumber, lalu memberi pseudo-label pada target.
# Student dilatih dari sumber + pseudo-label target, jadi mulai mengenal gaya bahasa target. Kuota per kelas
# mengikuti proporsi prediksi teacher. Kuota seragam terbukti memaksa kelas langka menjadi terlalu banyak.

# %%
def fit_predict(model_name, texts, labels, epochs, seed):
    """Kembalikan probabilitas untuk penjaga dan target."""
    if not USE_PRETRAINED:
        vector = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        model = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced")
        model.fit(vector.fit_transform(texts), labels)
        return [model.predict_proba(vector.transform(t)) for t in (guard_texts, target_texts)]
    weights = len(labels) / (len(CLASSES) * np.bincount(labels, minlength=len(CLASSES)))
    return train_predict(model_name, texts, labels, [guard_texts, target_texts], num_labels=len(CLASSES),
                         class_weights=weights, epochs=epochs, seed=seed)


def select_pseudo(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kuota mengikuti proporsi prediksi saat ini, karena proporsi asli target tidak diketahui."""
    pred, confidence = probs.argmax(1), probs.max(1)
    budget = PSEUDO_TOP_FRACTION * len(probs)
    shares = np.bincount(pred, minlength=probs.shape[1]) / len(probs)
    picked = []
    for cls in range(probs.shape[1]):
        members = np.where((pred == cls) & (confidence >= PSEUDO_MIN_CONFIDENCE))[0]
        picked.append(members[np.argsort(-confidence[members])][: int(budget * shares[cls])])
    index = np.concatenate(picked)
    return index, pred[index]


n_models = len(SETTINGS["model_runs"])
guard_teacher, target_teacher = np.zeros_like(guard_tfidf), np.zeros_like(target_tfidf)
guard_student, target_student = np.zeros_like(guard_tfidf), np.zeros_like(target_tfidf)
audit_rows = []
with stage("self_training"):
    for model_name, seed in SETTINGS["model_runs"]:
        g_t, t_t = fit_predict(model_name, fit_texts, y_fit, SETTINGS["epochs"], seed)
        guard_teacher += g_t / n_models
        target_teacher += t_t / n_models
        g_s, t_s = g_t, t_t
        if ALLOW_TARGET_TRAINING:
            for _ in range(SETTINGS["rounds"]):
                index, pseudo = select_pseudo(t_s)
                g_s, t_s = fit_predict(
                    model_name, fit_texts + [target_texts[i] for i in index],
                    np.concatenate([y_fit, pseudo]), SETTINGS["student_epochs"], seed,
                )
            audit_rows.append(pd.DataFrame({"index": index, "pseudo": pseudo}))
        guard_student += g_s / n_models
        target_student += t_s / n_models
record("teacher_sumber_saja", guard_teacher, target_teacher)
record("student_self_training", guard_student, target_student)

# %% [markdown]
# ## 9. Post-processing
#
# Student diterima hanya kalau skor penjaga di sumber naik minimal `MIN_GUARD_GAIN` dan tidak ada kelas yang
# hilang dari prediksi. Kalau tidak terbukti naik, teacher dipakai karena lebih sederhana. Pemilihan ini tidak memakai label target. Decision layer sengaja tidak
# dipakai, karena pengali yang difit di sumber belum tentu berlaku di target.

# %%
guard_gain = macro_f1(y_guard, guard_student.argmax(1)) - macro_f1(y_guard, guard_teacher.argmax(1))
all_classes = len(np.unique(target_student.argmax(1))) == len(CLASSES)
use_student = guard_gain >= MIN_GUARD_GAIN and all_classes and ALLOW_TARGET_TRAINING
final_probs = target_student if use_student else target_teacher
final_guard = guard_student if use_student else guard_teacher
record("final", final_guard, final_probs)
print(f"kenaikan skor penjaga: {guard_gain:+.4f} -> pakai {'student' if use_student else 'teacher'}")

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Kualitas pseudo-label** dibanding label audit, hanya untuk latihan ini.

# %%
if audit_rows:
    audit = pd.concat(audit_rows)
    audit["benar"] = audit["pseudo"].to_numpy() == y_target[audit["index"]]
    audit["kelas"] = [CLASSES[i] for i in audit["pseudo"]]
    summary_audit = audit.groupby("kelas").agg(jumlah=("benar", "size"), kecocokan=("benar", "mean")).round(3)
else:
    summary_audit = pd.DataFrame({"catatan": ["Target tidak dipakai saat training."]})
summary_audit

# %% [markdown]
# **Confusion matrix audit** di target.

# %%
plot_confusion(y_target, final_probs.argmax(1), CLASSES, "Confusion matrix audit target")

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
submission = pd.DataFrame({"id": target.index, "label": [CLASSES[i] for i in final_probs.argmax(1)]})
submission.to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("domain_shift", len(source) + len(target),
            {"auc_adversarial": round(shift_auc, 3),
             **{r["model"]: {"penjaga": r["penjaga_sumber"], "audit": r["audit_target"]} for r in results}},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# ISI_SETELAH_EDA
