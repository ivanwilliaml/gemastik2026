# %% [markdown]
# # 14 Regresi dan Ordinal
#
# Target berupa angka atau tingkatan. Contoh: rating 1-5 ulasan Tokopedia.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Memprediksi nilai atau tingkatan |
# | **Unit prediksi** | Satu ulasan mendapat nilai 1 sampai 5 |
# | **Metrik** | RMSE, MAE, dan QWK |
# | **Masalah terbesar** | Rating sangat timpang (bintang 5 sebanyak 73,4%) dan isi ulasan sering tidak sesuai bintangnya. |
# | **Data contoh** | 5.000 ulasan Tokopedia dari 1.396 produk |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** target berupa angka kontinu (RMSE, MAE) atau tingkatan berurutan (QWK), misalnya rating,
# skor kepuasan, atau tingkat urgensi keluhan.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `TEXT_COL`, `TARGET_COL`.
# - Metrik utama di bagian 6. Kalau kontinu, bagian cut point di bagian 9 dilewati.

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
TEXT_COL, TARGET_COL = "text", "rating"

SETTINGS = {
    "cepat": {"folds": 3, "epochs": 3, "model_runs": [("indobert", SEED)]},
    "penuh": {"folds": 5, "epochs": 4,
              "model_runs": [("indobert", SEED), ("indobertweet", SEED), ("indobert", SEED + 1)]},
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
# Muat data dan sisihkan holdout. Produk yang sama dijadikan satu grup, supaya ulasan satu produk tidak
# lintas fold.

# %%
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("tokopedia-product-reviews-2019.csv")).dropna(subset=[TEXT_COL, TARGET_COL])
    return frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)


data = load_data()
splitter = GroupShuffleSplit(1, test_size=HOLDOUT_SHARE, random_state=SEED)
fit_idx, hold_idx = next(splitter.split(data, groups=data["product_id"]))
train, holdout = data.iloc[fit_idx].reset_index(drop=True), data.iloc[hold_idx].reset_index(drop=True)
print(f"latih={len(train)} | holdout={len(holdout)} | produk unik={data['product_id'].nunique()}")
text_profile(train[TEXT_COL])

# %% [markdown]
# Sebaran target dan hubungan panjang teks dengan rating.

# %%
fig, axes = plt.subplots(1, 2)
train[TARGET_COL].value_counts().sort_index().plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set(title="Sebaran rating", xlabel="rating", ylabel="jumlah baris")
axes[0].tick_params(axis="x", rotation=0)
train.groupby(TARGET_COL)[TEXT_COL].apply(lambda s: s.str.split().str.len().median()).plot.bar(ax=axes[1], color="#4C72B0")
axes[1].set(title="Panjang ulasan per rating", xlabel="rating", ylabel="median jumlah kata")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
print("proporsi rating:", train[TARGET_COL].value_counts(normalize=True).sort_index().round(3).to_dict())

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Satu produk muncul di banyak ulasan | Grup validasi memakai id produk dan near-duplicate |
# | Median hanya 6 kata | Pembersihan ringan, dan tanda baca tetap ada |
# | Bintang 1 dan 2 masing-masing di bawah 2% | Cut point difit nested untuk QWK |

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan. Grup untuk validasi memakai id produk sekaligus near-duplicate, karena banyak ulasan
# pendek yang sama persis.

# %%
train["clean"] = train[TEXT_COL].map(clean_text)
holdout["clean"] = holdout[TEXT_COL].map(clean_text)
near_dup = near_duplicate_groups(train["clean"].tolist())
train["group"] = pd.factorize(train["product_id"].astype(str) + "_" + pd.Series(near_dup).astype(str))[0]
y = train[TARGET_COL].to_numpy()
y_holdout = holdout[TARGET_COL].to_numpy()
print(f"grup near-duplicate: {len(set(near_dup))} dari {len(train)} baris")

# %% [markdown]
# ## 6. Validasi
#
# Skema `StratifiedGroupKFold`. Ada tiga metrik: RMSE dan MAE untuk nilai kontinu, dan QWK untuk tingkatan.
# QWK butuh prediksi bulat, jadi cut point ikut menentukan skornya.

# %%
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, np.clip(np.rint(y_pred), 1, 5).astype(int), weights="quadratic"))


assert abs(rmse([1, 3], [1, 1]) - np.sqrt(2)) < 1e-9
assert qwk([1, 2, 3], [1, 2, 3]) == 1.0
folds = make_folds(y, train["group"], SETTINGS["folds"])
print(pd.crosstab(folds, y))


def record_reg(name: str, oof_pred, holdout_pred) -> None:
    results.append(
        {"model": name, "rmse_oof": round(rmse(y, oof_pred), 4), "mae_oof": round(mean_absolute_error(y, oof_pred), 4),
         "qwk_oof": round(qwk(y, oof_pred), 4), "qwk_ci95": bootstrap_interval(y, oof_pred, qwk),
         "rmse_holdout": round(rmse(y_holdout, holdout_pred), 4), "qwk_holdout": round(qwk(y_holdout, holdout_pred), 4)}
    )
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Tebak rata-rata, lalu TF-IDF + Ridge.

# %%
from sklearn.linear_model import Ridge  # noqa: E402

record_reg("rata_rata", np.full(len(y), y.mean()), np.full(len(y_holdout), y.mean()))

oof_ridge, hold_ridge = np.zeros(len(y)), np.zeros(len(y_holdout))
for fold in np.unique(folds):
    fit, val = folds != fold, folds == fold
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    model = Ridge(alpha=1.0).fit(vectorizer.fit_transform(train.loc[fit, "clean"]), y[fit])
    oof_ridge[val] = model.predict(vectorizer.transform(train.loc[val, "clean"]))
    hold_ridge += model.predict(vectorizer.transform(holdout["clean"])) / SETTINGS["folds"]
record_reg("tfidf_ridge", oof_ridge, hold_ridge)

# %% [markdown]
# ## 8. Model Utama
#
# Transformer dengan satu keluaran dan loss MSE. Target distandarkan dulu, supaya loss tidak terlalu besar
# di awal training.

# %%
mean_y, std_y = y.mean(), y.std()
with stage("transformer"):
    if USE_PRETRAINED:
        oof_scaled, hold_scaled = cv_transformer(
            train["clean"], (y - mean_y) / std_y, holdout["clean"], folds,
            SETTINGS["model_runs"], SETTINGS["epochs"], problem="mse",
        )
        oof_main = oof_scaled[:, 0] * std_y + mean_y
        hold_main = hold_scaled[:, 0] * std_y + mean_y
    else:
        oof_main, hold_main = oof_ridge, hold_ridge
record_reg("model_utama", oof_main, hold_main)

# %% [markdown]
# ## 9. Post-processing
#
# **Cut point untuk QWK.** Membulatkan langsung biasanya rugi, karena prediksi regresi menumpuk di tengah.
# Batas antar rating dicari untuk memaksimalkan QWK. Skor OOF dihitung nested.

# %%
def apply_cuts(scores: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(cuts)) + 1


def fit_cuts(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Coordinate ascent pada empat batas."""
    cuts = np.array([1.5, 2.5, 3.5, 4.5])
    grid = np.quantile(scores, np.linspace(0.005, 0.995, 80))
    for _ in range(3):
        for i in range(len(cuts)):
            values = []
            for candidate in grid:
                trial = cuts.copy()
                trial[i] = candidate
                values.append(cohen_kappa_score(target, apply_cuts(scores, trial), weights="quadratic"))
            cuts[i] = grid[int(np.argmax(values))]
    return np.sort(cuts)


oof_cut = np.zeros(len(y))
with stage("cut_point"):
    for fold in np.unique(folds):
        fit = folds != fold
        oof_cut[~fit] = apply_cuts(oof_main[~fit], fit_cuts(oof_main[fit], y[fit]))
    cuts = fit_cuts(oof_main, y)
hold_cut = apply_cuts(hold_main, cuts)
record_reg("model_utama_plus_cut_point", oof_cut, hold_cut)
print("cut point:", cuts.round(2).tolist())

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Prediksi dibanding nilai asli.** Lihat apakah prediksi menumpuk di rating mayoritas.

# %%
fig, ax = plt.subplots()
ax.boxplot([oof_main[y == r] for r in range(1, 6)], labels=range(1, 6), showfliers=False)
ax.set(title="Prediksi OOF per rating asli", xlabel="rating asli", ylabel="prediksi")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Confusion matrix** setelah cut point.

# %%
plot_confusion(y, oof_cut.astype(int), [1, 2, 3, 4, 5], "Confusion matrix OOF")

# %% [markdown]
# **Salah paling jauh.** Biasanya kelihatan ulasan yang isinya tidak sesuai rating.

# %%
errors = train.assign(prediksi=oof_main.round(2), selisih=np.abs(oof_main - y).round(2))
errors.sort_values("selisih", ascending=False)[["clean", TARGET_COL, "prediksi"]].head(8)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": holdout.index, "rating": hold_cut.astype(int), "skor": hold_main.round(3)}).to_csv(
    RESULT_DIR / "submission.csv", index=False
)
pd.DataFrame({"skor": oof_main, "fold": folds, "y": y}).to_csv(RESULT_DIR / "oof.csv", index=False)
run_summary("regresi_ordinal", len(data),
            {r["model"]: {"qwk_oof": r["qwk_oof"], "qwk_holdout": r["qwk_holdout"], "rmse_holdout": r["rmse_holdout"]}
             for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - QWK OOF naik dari TF-IDF 0,3926 ke model utama 0,6047, lalu 0,6354 dengan cut point. Holdout 0,5815.
# - **Cut point bergantung pada metrik.** QWK dan MAE membaik, tapi RMSE memburuk (0,568 ke 0,6535). Kalau metriknya RMSE, jangan dibulatkan.
# - Kesalahan terbesar berasal dari label berisik, misalnya ulasan `kurang bagus. kaya bekas` diberi bintang 5.
