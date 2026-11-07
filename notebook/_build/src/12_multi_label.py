# %% [markdown]
# # 12 Klasifikasi Multi-Label
#
# Satu teks bisa punya beberapa label sekaligus. Contoh: jenis ujaran kebencian di Twitter.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** satu teks bisa masuk lebih dari satu kategori, misalnya komentar MBG yang membahas
# korupsi sekaligus keracunan. Metriknya biasanya macro-F1 atau micro-F1 per label.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `TEXT_COL`, `LABEL_COLS`.
# - Kalau tiap label punya polaritas, misalnya positif atau negatif per aspek, pakai folder `13`.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya TF-IDF, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
HOLDOUT_SHARE = 0.2
TEXT_COL = "Tweet"
LABEL_COLS = ["Abusive", "HS_Individual", "HS_Group", "HS_Religion", "HS_Race",
              "HS_Physical", "HS_Gender", "HS_Other"]

SETTINGS = {
    "cepat": {"folds": 3, "epochs": 4, "model_runs": [("indobertweet", SEED)]},
    "penuh": {"folds": 5, "epochs": 5,
              "model_runs": [("indobertweet", SEED), ("indobert", SEED), ("indobertweet", SEED + 1)]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include clf_common

# %% include cv_models

# %% include transformer_seqcls

# %% [markdown]
# ## 4. EDA
#
# Muat data dan sisihkan holdout yang hanya dipakai untuk laporan akhir.

# %%
from sklearn.model_selection import train_test_split  # noqa: E402


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("data.csv"), encoding="latin-1")
    frame = frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)
    return frame[[TEXT_COL] + LABEL_COLS]


data = load_data()
train, holdout = train_test_split(data, test_size=HOLDOUT_SHARE, random_state=SEED)
train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)
print(f"latih={len(train)} | holdout={len(holdout)}")
text_profile(train[TEXT_COL])

# %% [markdown]
# Jumlah positif per label, jumlah label per baris, dan label mana yang sering muncul bersama.

# %%
prevalence = train[LABEL_COLS].mean().round(3)
per_row = train[LABEL_COLS].sum(axis=1)
print("proporsi positif per label:", prevalence.to_dict())
print("jumlah label per baris:", per_row.value_counts().sort_index().to_dict())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
prevalence.sort_values().plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set(title="Proporsi positif per label", xlabel="proporsi", ylabel="label")
correlation = train[LABEL_COLS].corr().to_numpy()
image = axes[1].imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
ticks = range(len(LABEL_COLS))
axes[1].set(xticks=ticks, yticks=ticks, xticklabels=LABEL_COLS, yticklabels=LABEL_COLS,
            title="Korelasi antar label", xlabel="label", ylabel="label")
axes[1].tick_params(axis="x", rotation=90)
axes[1].grid(False)
fig.colorbar(image)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Temuan dan keputusan**
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 5. Preprocessing
#
# Token anonimisasi dibuang, kata kasar tetap dipertahankan karena menjadi sinyal utama.

# %%
TOKEN_RE = re.compile(r"\b(USER|RT|URL)\b|\\n")
BYTES_RE = re.compile(r"(?:\\x[0-9a-f]{2})+")


def decode_bytes(match: re.Match) -> str:
    """Data ini menyimpan emoji sebagai teks byte, misalnya \\xf0\\x9f\\x98\\xa0. Kembalikan jadi emoji."""
    return " " + bytes.fromhex(match.group(0).replace("\\x", "")).decode("utf-8", "ignore") + " "


def preprocess(text: str) -> str:
    return clean_text(TOKEN_RE.sub(" ", BYTES_RE.sub(decode_bytes, str(text))))


train["clean"] = train[TEXT_COL].map(preprocess)
holdout["clean"] = holdout[TEXT_COL].map(preprocess)
train["group"] = near_duplicate_groups(train["clean"].tolist())
y = train[LABEL_COLS].to_numpy()
y_holdout = holdout[LABEL_COLS].to_numpy()
print(f"grup near-duplicate: {train['group'].nunique()} dari {len(train)} baris")

# %% [markdown]
# ## 6. Validasi
#
# Fold distratifikasi menurut jumlah label per baris, dan grup near-duplicate tidak lintas fold. Metrik
# utama macro-F1 antar label, dengan micro-F1 sebagai pembanding.

# %%
def macro_f1_labels(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def micro_f1_labels(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


_true = np.array([[1, 0], [0, 1]])
assert macro_f1_labels(_true, _true) == 1.0
assert abs(micro_f1_labels(_true, np.array([[1, 1], [0, 1]])) - 0.8) < 1e-9

folds = make_folds(np.minimum(y.sum(1), 3), train["group"], SETTINGS["folds"])
print("baris per fold:", np.bincount(folds).tolist())

# %% [markdown]
# ## 7. Baseline
#
# Tebak semua negatif sebagai batas bawah, lalu TF-IDF + regresi logistik per label dengan threshold 0,5.

# %%
record("semua_negatif", np.zeros_like(y), np.zeros_like(y_holdout), macro_f1_labels)
with stage("baseline_tfidf"):
    oof_tfidf, hold_tfidf = cv_tfidf(train["clean"], y, holdout["clean"], folds, multilabel=True)
record("tfidf_lr", (oof_tfidf >= 0.5).astype(int), (hold_tfidf >= 0.5).astype(int), macro_f1_labels)

# %% [markdown]
# ## 8. Model Utama
#
# Satu transformer dengan satu keluaran per label, jadi model bisa memanfaatkan hubungan antar label.
# Loss BCE memakai `pos_weight`, karena tanpa itu label langka seperti `HS_Gender` (2%) tidak terlatih.

# %%
with stage("transformer"):
    if USE_PRETRAINED:
        oof_main, hold_main = cv_transformer(
            train["clean"], y, holdout["clean"], folds, SETTINGS["model_runs"], SETTINGS["epochs"],
            problem="bce",
        )
    else:
        oof_main, hold_main = oof_tfidf, hold_tfidf
record("model_utama", (oof_main >= 0.5).astype(int), (hold_main >= 0.5).astype(int), macro_f1_labels)

# %% [markdown]
# ## 9. Post-processing
#
# Threshold per label, karena label langka hampir tidak pernah mencapai 0,5. Skor OOF dihitung nested.
# Tiga kandidat dibandingkan: TF-IDF, transformer, dan rata-rata keduanya. Yang dipakai adalah kandidat dengan
# skor OOF nested tertinggi.

# %%
def label_thresholds(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    grid = np.arange(0.03, 0.91, 0.01)
    best = []
    for col in range(target.shape[1]):
        values = [f1_score(target[:, col], scores[:, col] >= t, zero_division=0) for t in grid]
        best.append(grid[int(np.argmax(values))])
    return np.array(best)


def nested_thresholded(scores: np.ndarray) -> np.ndarray:
    output = np.zeros_like(y)
    for fold in np.unique(folds):
        fit = folds != fold
        output[~fit] = scores[~fit] >= label_thresholds(scores[fit], y[fit])
    return output


candidates = {"tfidf": (oof_tfidf, hold_tfidf), "transformer": (oof_main, hold_main),
              "rata_rata": ((oof_tfidf + oof_main) / 2, (hold_tfidf + hold_main) / 2)}
scored = {}
for name, (oof_scores, hold_scores) in candidates.items():
    scored[name] = nested_thresholded(oof_scores)
    record(f"{name}_plus_threshold", scored[name], (hold_scores >= label_thresholds(oof_scores, y)).astype(int), macro_f1_labels)

best_name = max(scored, key=lambda n: macro_f1_labels(y, scored[n]))
best_oof, best_hold = candidates[best_name]
oof_decision = scored[best_name]
thresholds = label_thresholds(best_oof, y)
hold_decision = (best_hold >= thresholds).astype(int)
print("dipilih dari OOF:", best_name, "| micro-F1 OOF:", round(micro_f1_labels(y, oof_decision), 4))

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Skor per label.** Label langka biasanya paling lemah, dan bisa jadi datanya memang tidak cukup.

# %%
per_label = pd.DataFrame(
    {"positif_latih": y.sum(0), "threshold": thresholds.round(2),
     "f1_oof": f1_score(y, oof_decision, average=None, zero_division=0).round(3),
     "f1_holdout": f1_score(y_holdout, hold_decision, average=None, zero_division=0).round(3)},
    index=LABEL_COLS,
).sort_values("positif_latih", ascending=False)
per_label

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame(hold_decision, columns=LABEL_COLS).rename_axis("id").to_csv(RESULT_DIR / "submission.csv")
pd.DataFrame(best_oof, columns=LABEL_COLS).assign(fold=folds).to_csv(RESULT_DIR / "oof.csv", index=False)
run_summary("multi_label", len(data), {r["model"]: {"oof": r["oof"], "holdout": r["holdout"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# ISI_SETELAH_EDA
