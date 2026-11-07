# %% [markdown]
# # 09 Klasifikasi Biner
#
# Dua kelas, misalnya hoaks atau bukan, toxic atau bukan. Contoh: ujaran kebencian di Twitter.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** target hanya dua kelas dan semua data latih berlabel. Metriknya biasanya F1 kelas
# positif atau AUC. Kalau labelnya sedikit, pakai folder `08`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `TEXT_COL`, `LABEL_COL`.
# - `score_fn` di bagian 6 kalau metriknya lain. AUC tidak butuh threshold.
# - `MODEL_RUNS` di config. `indobertweet` untuk media sosial, `indobert` untuk teks formal.

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
TEXT_COL, LABEL_COL = "Tweet", "HS"
MIN_GAIN = 0.002  # threshold hanya dipakai kalau skor OOF nested naik lebih dari ini

SETTINGS = {
    "cepat": {"folds": 3, "epochs": 3, "model_runs": [("indobertweet", SEED)]},
    "penuh": {"folds": 5, "epochs": 4,
              "model_runs": [("indobertweet", SEED), ("indobert", SEED), ("indobertweet", SEED + 1)]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include clf_common

# %% include cv_models

# %% include transformer_seqcls

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat data, lalu sisihkan holdout. Holdout berperan seperti test tersembunyi dan hanya dipakai untuk
# laporan akhir.

# %%
from sklearn.metrics import roc_auc_score as auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("data.csv"), encoding="latin-1")
    frame = frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)
    return frame[[TEXT_COL, LABEL_COL]]


data = load_data()
train, holdout = train_test_split(
    data, test_size=HOLDOUT_SHARE, stratify=data[LABEL_COL], random_state=SEED
)
train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)
print(f"latih={len(train)} | holdout={len(holdout)}")
print("proporsi positif:", round(train[LABEL_COL].mean(), 3))
text_profile(train[TEXT_COL])

# %% [markdown]
# Token khusus yang sering muncul, dan beda panjang teks antar kelas.

# %%
token_share = {t: round(100 * train[TEXT_COL].str.contains(rf"\b{t}\b").mean(), 1)
               for t in ["USER", "RT", "URL"]}
print("persen baris berisi token khusus:", token_share)
print("AUC adversarial latih vs holdout:", round(adversarial_auc(train[TEXT_COL], holdout[TEXT_COL]), 3))

lengths = train[TEXT_COL].str.split().str.len().clip(upper=60)
fig, ax = plt.subplots()
for value, name in ((0, "negatif"), (1, "positif")):
    lengths[train[LABEL_COL] == value].plot.hist(bins=30, alpha=0.6, ax=ax, label=name)
ax.set(title="Panjang teks per kelas", xlabel="jumlah kata", ylabel="jumlah baris")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Temuan dan keputusan**
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 5. Preprocessing
#
# Token `USER`, `RT`, dan `URL` dibuang karena hanya sisa anonimisasi. Kata kasar dan emoji tetap
# dipertahankan karena justru menjadi sinyal utama. Tanda baca tidak dibuang.

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
y = train[LABEL_COL].to_numpy()
y_holdout = holdout[LABEL_COL].to_numpy()
print(f"grup near-duplicate: {train['group'].nunique()} dari {len(train)} baris")

# %% [markdown]
# ## 6. Validasi
#
# Skema `StratifiedGroupKFold`. Metrik utama F1 kelas positif, dan AUC ikut dilaporkan karena tidak
# bergantung pada threshold.

# %%
def f1_positive(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))


assert abs(f1_positive([0, 1, 1, 1], [0, 1, 1, 0]) - 0.8) < 1e-9

folds = make_folds(y, train["group"], SETTINGS["folds"])
print(pd.crosstab(folds, y))

# %% [markdown]
# ## 7. Baseline
#
# Tebak semua positif, lalu TF-IDF + regresi logistik.

# %%
record("semua_positif", np.ones(len(y), dtype=int), np.ones(len(y_holdout), dtype=int), f1_positive)
with stage("baseline_tfidf"):
    oof_tfidf, hold_tfidf = cv_tfidf(train["clean"], y, holdout["clean"], folds)
record("tfidf_lr", oof_tfidf.argmax(1), hold_tfidf.argmax(1), f1_positive)
print("AUC OOF tfidf:", round(auc_score(y, oof_tfidf[:, 1]), 4))

# %% [markdown]
# ## 8. Model Utama
#
# Transformer dengan cross-validation. Kalau `USE_PRETRAINED=False`, model utama sama dengan baseline.

# %%
with stage("transformer"):
    if USE_PRETRAINED:
        oof_main, hold_main = cv_transformer(
            train["clean"], y, holdout["clean"], folds, SETTINGS["model_runs"], SETTINGS["epochs"]
        )
    else:
        oof_main, hold_main = oof_tfidf, hold_tfidf
record("model_utama", oof_main.argmax(1), hold_main.argmax(1), f1_positive)
print("AUC OOF model utama:", round(auc_score(y, oof_main[:, 1]), 4))

# %% [markdown]
# ## 9. Post-processing
#
# Threshold untuk F1. Skor OOF dihitung nested: threshold tiap fold difit dari fold lain. Threshold
# untuk holdout difit dari seluruh OOF, dan hanya dipakai kalau skor nested naik lebih dari `MIN_GAIN`.

# %%
def best_threshold(scores: np.ndarray, target: np.ndarray) -> float:
    grid = np.arange(0.1, 0.91, 0.01)
    values = [f1_positive(target, scores >= t) for t in grid]
    return float(grid[int(np.argmax(values))])


scores = oof_main[:, 1]
nested_pred = np.zeros(len(y), dtype=int)
for fold in np.unique(folds):
    fit = folds != fold
    nested_pred[~fit] = scores[~fit] >= best_threshold(scores[fit], y[fit])
threshold = best_threshold(scores, y)
hold_threshold = (hold_main[:, 1] >= threshold).astype(int)
record("model_utama_plus_threshold", nested_pred, hold_threshold, f1_positive)

use_threshold = f1_positive(y, nested_pred) > f1_positive(y, oof_main.argmax(1)) + MIN_GAIN
oof_decision = nested_pred if use_threshold else oof_main.argmax(1)
hold_pred = hold_threshold if use_threshold else hold_main.argmax(1)
print(f"threshold {threshold:.2f} -> dipakai: {use_threshold}")

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Confusion matrix OOF.**

# %%
plot_confusion(y, oof_decision, ["negatif", "positif"], "Confusion matrix OOF")

# %% [markdown]
# **Salah prediksi paling yakin.** Di sini biasanya kelihatan label yang meragukan atau kata kasar yang
# tidak bermaksud menyerang.

# %%
errors = train.assign(prediksi=oof_decision, skor=scores.round(3))
errors = errors[errors[LABEL_COL] != errors["prediksi"]]
errors["yakin"] = (errors["skor"] - 0.5).abs()
errors.sort_values("yakin", ascending=False)[["clean", LABEL_COL, "skor"]].head(8)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": holdout.index, "label": hold_pred}).to_csv(RESULT_DIR / "submission.csv", index=False)
pd.DataFrame({"skor": scores, "fold": folds, "y": y}).to_csv(RESULT_DIR / "oof.csv", index=False)
run_summary("biner", len(data), {r["model"]: {"oof": r["oof"], "holdout": r["holdout"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# ISI_SETELAH_EDA
