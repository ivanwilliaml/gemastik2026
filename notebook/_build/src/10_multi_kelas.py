# %% [markdown]
# # 10 Klasifikasi Multi-Kelas dengan Imbalance
#
# Tiga kelas atau lebih dengan proporsi timpang. Contoh: sentimen komentar YouTube tentang MBG.
#
# **Peringatan data:** label di dataset ini **hasil keluaran model**, bukan anotasi manusia. Skornya
# mengukur seberapa mirip dengan model itu, bukan kebenaran sentimen. Notebook ini tetap berguna
# sebagai pola, dan kolom `Confidence Score` dipakai untuk melihat dampak label yang meragukan.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Klasifikasi multi-kelas dengan kelas timpang |
# | **Unit prediksi** | Satu komentar mendapat satu kelas sentimen |
# | **Metrik** | Macro-F1 |
# | **Masalah terbesar** | Kelas timpang dan label bukan buatan manusia: negatif 69,1%, netral 8,4%. Label dataset ini tampaknya keluaran model, jadi skornya mengukur kemiripan dengan model itu. |
# | **Data contoh** | 5.000 komentar YouTube MBG |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** semua data latih berlabel, kelasnya tiga atau lebih, dan metriknya macro-F1. Berlaku
# untuk sentimen, emosi, stance, atau kategori topik.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `TEXT_COL`, `LABEL_COL`, `CLASSES`.
# - Sel analisis `Confidence Score` di bagian 10 dihapus kalau kolomnya tidak ada.
# - Varian emosi berlabel manusia ada di paket offline: `indonlu/dataset/emot_emotion-twitter`.

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
TEXT_COL, LABEL_COL, CONF_COL = "Cleaned Comment", "Sentiment", "Confidence Score"
CLASSES = ["negative", "neutral", "positive"]
MIN_GAIN = 0.002  # pengali hanya dipakai kalau skor OOF nested naik lebih dari ini

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

# %% include decision_multiclass

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat data, lalu sisihkan holdout yang hanya dipakai untuk laporan akhir.

# %%
from sklearn.model_selection import train_test_split  # noqa: E402


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("comment_sentiment.csv"))
    frame = frame.dropna(subset=[TEXT_COL, LABEL_COL])
    return frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)


data = load_data()
train, holdout = train_test_split(
    data, test_size=HOLDOUT_SHARE, stratify=data[LABEL_COL], random_state=SEED
)
train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)
print(f"latih={len(train)} | holdout={len(holdout)}")
text_profile(train[TEXT_COL])

# %% [markdown]
# Distribusi kelas dan sebaran keyakinan label.

# %%
fig, axes = plt.subplots(1, 2)
train[LABEL_COL].value_counts().reindex(CLASSES).plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set(title="Distribusi kelas", xlabel="kelas", ylabel="jumlah baris")
axes[0].tick_params(axis="x", rotation=0)
train[CONF_COL].plot.hist(bins=30, ax=axes[1], color="#4C72B0")
axes[1].set(title="Keyakinan label", xlabel="confidence score", ylabel="jumlah baris")
plt.tight_layout()
plt.show()
print("proporsi kelas:", train[LABEL_COL].value_counts(normalize=True).round(3).to_dict())
print("keyakinan median per kelas:", train.groupby(LABEL_COL)[CONF_COL].median().round(3).to_dict())

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Netral hanya 8,4% | Pakai macro-F1, bukan akurasi. Tebak mayoritas akurasinya tinggi tapi macro-F1 0,2724 |
# | Keyakinan label median netral 0,815, negatif 0,988 | Label netral paling meragukan, jadi laporkan skor per keyakinan |
# | Ada komentar berulang | Grup near-duplicate tidak boleh lintas fold |
# | Maksimum 1.485 kata | Teks dipotong oleh tokenizer, dan pembersihan tetap ringan |

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan, emoji dan tanda baca dipertahankan. Slang tidak dinormalisasi karena IndoBERTweet
# dilatih dari teks informal. Grup near-duplicate dihitung untuk validasi.

# %%
class_to_id = {name: i for i, name in enumerate(CLASSES)}
train["clean"] = train[TEXT_COL].map(clean_text)
holdout["clean"] = holdout[TEXT_COL].map(clean_text)
train["group"] = near_duplicate_groups(train["clean"].tolist())
y = train[LABEL_COL].map(class_to_id).to_numpy()
y_holdout = holdout[LABEL_COL].map(class_to_id).to_numpy()
print(f"grup near-duplicate: {train['group'].nunique()} dari {len(train)} baris")

# %% [markdown]
# ## 6. Validasi
#
# Skema `StratifiedGroupKFold`, metrik macro-F1. Accuracy tidak dipakai karena tebak mayoritas saja sudah
# terlihat tinggi saat kelasnya timpang.

# %%
folds = make_folds(y, train["group"], SETTINGS["folds"])
print(pd.crosstab(folds, train[LABEL_COL]))

# %% [markdown]
# ## 7. Baseline
#
# Tebak mayoritas, lalu TF-IDF + regresi logistik berbobot kelas.

# %%
majority = np.bincount(y).argmax()
record("mayoritas", np.full(len(y), majority), np.full(len(y_holdout), majority))
with stage("baseline_tfidf"):
    oof_tfidf, hold_tfidf = cv_tfidf(train["clean"], y, holdout["clean"], folds)
record("tfidf_lr", oof_tfidf.argmax(1), hold_tfidf.argmax(1))

# %% [markdown]
# ## 8. Model Utama
#
# Transformer dengan cross-validation dan bobot kelas seimbang di loss. Tanpa bobot kelas, kelas minoritas
# cenderung tidak pernah ditebak. Di mode penuh, hasil tiga run dirata-rata.

# %%
with stage("transformer"):
    if USE_PRETRAINED:
        oof_main, hold_main = cv_transformer(
            train["clean"], y, holdout["clean"], folds, SETTINGS["model_runs"], SETTINGS["epochs"]
        )
    else:
        oof_main, hold_main = oof_tfidf, hold_tfidf
record("model_utama", oof_main.argmax(1), hold_main.argmax(1))

# %% [markdown]
# ## 9. Post-processing
#
# Pengali per kelas untuk macro-F1. Biasanya ini langkah paling murah dengan dampak paling besar saat
# kelasnya timpang. Skor OOF dihitung nested, jadi jujur.

# %%
oof_multiplier = nested_multiplier_predictions(oof_main, y, folds)
multipliers = fit_class_multipliers(oof_main, y)
hold_multiplier = (hold_main * multipliers).argmax(1)
record("model_utama_plus_pengali", oof_multiplier, hold_multiplier)

use_multiplier = macro_f1(y, oof_multiplier) > macro_f1(y, oof_main.argmax(1)) + MIN_GAIN
oof_decision = oof_multiplier if use_multiplier else oof_main.argmax(1)
hold_decision = hold_multiplier if use_multiplier else hold_main.argmax(1)
print("pengali:", dict(zip(CLASSES, multipliers.round(2).tolist())), "-> dipakai:", use_multiplier)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Confusion matrix OOF.** Lihat kelas minoritas ke mana salahnya.

# %%
plot_confusion(y, oof_decision, CLASSES, "Confusion matrix OOF")

# %% [markdown]
# **Skor menurut keyakinan label.** Kalau skor naik tajam di label yang yakin, sebagian error sebenarnya
# berasal dari label yang meragukan, bukan dari model.

# %%
buckets = pd.cut(train[CONF_COL], [0, 0.6, 0.8, 0.95, 1.0])
by_confidence = pd.DataFrame(
    {"keyakinan": buckets, "y": y, "pred": oof_decision}
).groupby("keyakinan", observed=True).apply(
    lambda g: pd.Series({"baris": len(g), "macro_f1": round(macro_f1(g["y"], g["pred"]), 3)}),
    include_groups=False,
)
by_confidence

# %% [markdown]
# **Salah prediksi paling yakin.**

# %%
errors = train.assign(prediksi=[CLASSES[i] for i in oof_decision], yakin=oof_main.max(1).round(3))
errors = errors[errors[LABEL_COL] != errors["prediksi"]]
errors.sort_values("yakin", ascending=False)[["clean", LABEL_COL, "prediksi", "yakin"]].head(8)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
submission = pd.DataFrame({"id": holdout.index, "label": [CLASSES[i] for i in hold_decision]})
submission.to_csv(RESULT_DIR / "submission.csv", index=False)
pd.DataFrame(oof_main, columns=CLASSES).assign(fold=folds, y=y).to_csv(RESULT_DIR / "oof.csv", index=False)
run_summary("multi_kelas", len(data), {r["model"]: {"oof": r["oof"], "holdout": r["holdout"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Macro-F1 OOF naik dari TF-IDF 0,6242 ke model utama 0,7438, lalu 0,7493 dengan pengali kelas. Holdout 0,7485.
# - Pengali difit nested dan hanya dipakai kalau naik minimal 0,002.
# - Macro-F1 pada keyakinan rendah hanya 0,431, sedangkan pada keyakinan tinggi 0,881. Kesalahan terbesar terkumpul di teks yang memang ambigu.
# - Contoh salah seperti `good job pak` berlabel negatif menunjukkan ada noise pada label.
