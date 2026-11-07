# %% [markdown]
# # 20 Kemiripan Teks dan Parafrase
#
# Mengukur seberapa mirip dua teks. Contoh: STS Benchmark terjemahan Indonesia dengan skor 0 sampai 5.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Mengukur kemiripan dua teks |
# | **Unit prediksi** | Satu pasangan teks mendapat skor 0 sampai 5 |
# | **Metrik** | Pearson dan Spearman |
# | **Masalah terbesar** | Sumber data holdout berbeda (`answer-answer` hanya ada di holdout), jadi OOF terlalu optimistis: cross-encoder OOF 0,8319 turun ke holdout 0,7669. |
# | **Data contoh** | STSB versi Indonesia, 1.500 latih, 1.379 holdout |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** data berisi pasangan teks dengan skor kemiripan, atau soal meminta deteksi parafrase. Metriknya
# biasanya Pearson, Spearman, atau F1. Kalau labelnya entailment, pakai folder `21`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, kolom `text_1`, `text_2`, `score`.
# - `PARAPHRASE_MIN_SCORE`, batas skor yang dianggap parafrase.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya fitur TF-IDF, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
PARAPHRASE_MIN_SCORE = 4.0

SETTINGS = {
    "cepat": {"folds": 3, "epochs": 3, "model_runs": [("indobert", SEED)]},
    "penuh": {"folds": 5, "epochs": 5, "model_runs": [("indobert", SEED), ("indobert", SEED + 1)]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include embedding

# %% include transformer_seqcls

# %% [markdown]
# ## 4. EDA
#
# Muat data. Split validation dipakai untuk latih dengan cross-validation, dan split test sebagai holdout.

# %%
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402


def load_data(split: str) -> pd.DataFrame:
    frame = pd.read_parquet(EXTRA_DATA_DIR / "stsb_id" / f"{split}.parquet")
    frame["score"] = pd.to_numeric(frame["correlation"], errors="coerce")
    return frame.dropna(subset=["text_1", "text_2", "score"]).head(N_ROWS).reset_index(drop=True)


train, test = load_data("validation"), load_data("test")
print(f"latih={len(train)} | holdout={len(test)}")
print("sumber data:", train["data"].value_counts().head(5).to_dict())
print("sumber holdout:", test["data"].value_counts().head(5).to_dict())
train[["text_1", "text_2", "score"]].head(3)

# %% [markdown]
# Sebaran skor dan pergeseran antara latih dan holdout.

# %%
fig, ax = plt.subplots()
ax.hist(train["score"], bins=20, alpha=0.6, label="latih", density=True)
ax.hist(test["score"], bins=20, alpha=0.6, label="holdout", density=True)
ax.set(title="Sebaran skor kemiripan", xlabel="skor", ylabel="kepadatan")
ax.legend()
plt.tight_layout()
plt.show()
shift_auc = adversarial_auc((train["text_1"] + " " + train["text_2"]).tolist(), (test["text_1"] + " " + test["text_2"]).tolist())
print(f"AUC adversarial: {shift_auc:.3f} | parafrase latih: {(train['score'] >= PARAPHRASE_MIN_SCORE).mean():.1%}")

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Satu teks muncul di banyak pasangan | Grup validasi dari kedua teks |
# | Skor berbeda jauh antar sumber | Laporkan Pearson per sumber |
# | AUC adversarial 0,508 | Dari isi teks tidak ada shift yang kuat, tapi skor per sumber tetap berbeda |

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan. Satu teks bisa muncul di banyak pasangan, jadi grup validasi dibuat dari kedua teks, supaya
# teks yang sama tidak lintas fold.

# %%
for frame in (train, test):
    frame["a"], frame["b"] = frame["text_1"].map(clean_text), frame["text_2"].map(clean_text)
all_text = pd.concat([train["a"], train["b"]]).unique()
text_to_id = {t: i for i, t in enumerate(all_text)}
parent = list(range(len(all_text)))


def find_root(i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for a, b in zip(train["a"], train["b"]):
    parent[find_root(text_to_id[a])] = find_root(text_to_id[b])
train["group"] = [find_root(text_to_id[a]) for a in train["a"]]
y, y_holdout = train["score"].to_numpy(), test["score"].to_numpy()
print(f"grup: {train['group'].nunique()} dari {len(train)} pasangan")

# %% [markdown]
# ## 6. Validasi
#
# `GroupKFold` berdasarkan grup teks. Metrik utama Pearson dan Spearman. F1 parafrase dihitung setelah skor diberi
# threshold.

# %%
def pearson(y_true, y_pred) -> float:
    return float(pearsonr(y_true, y_pred)[0])


assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
folds = np.zeros(len(train), dtype=int)
for fold, (_, val_idx) in enumerate(GroupKFold(SETTINGS["folds"]).split(train, y, train["group"])):
    folds[val_idx] = fold
results: list[dict] = []


def record(name: str, oof, holdout) -> None:
    results.append({"model": name, "pearson_oof": round(pearson(y, oof), 4),
                    "spearman_oof": round(float(spearmanr(y, oof)[0]), 4),
                    "pearson_holdout": round(pearson(y_holdout, holdout), 4),
                    "spearman_holdout": round(float(spearmanr(y_holdout, holdout)[0]), 4)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Cosine TF-IDF, tanpa pelatihan. Lalu Ridge dari fitur tumpang tindih kata.

# %%
pair_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pd.concat([train["a"], train["b"]]))


def cosine_tfidf(frame: pd.DataFrame) -> np.ndarray:
    left, right = pair_vectorizer.transform(frame["a"]), pair_vectorizer.transform(frame["b"])
    return np.asarray(left.multiply(right).sum(1)).ravel()


def pair_features(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for a, b, cosine in zip(frame["a"], frame["b"], cosine_tfidf(frame)):
        set_a, set_b = set(a.split()), set(b.split())
        rows.append([cosine, len(set_a & set_b) / max(len(set_a | set_b), 1),
                     abs(len(set_a) - len(set_b)) / max(len(set_a | set_b), 1)])
    return np.array(rows)


x_train, x_test = pair_features(train), pair_features(test)
record("cosine_tfidf", x_train[:, 0], x_test[:, 0])
oof_ridge, hold_ridge = np.zeros(len(y)), np.zeros(len(y_holdout))
for fold in range(SETTINGS["folds"]):
    fit, val = folds != fold, folds == fold
    model = Ridge(alpha=1.0).fit(x_train[fit], y[fit])
    oof_ridge[val] = model.predict(x_train[val])
    hold_ridge += model.predict(x_test) / SETTINGS["folds"]
record("ridge_fitur_kata", oof_ridge, hold_ridge)

# %% [markdown]
# ## 8. Model Utama
#
# **Bi-encoder e5:** cosine embedding, tanpa pelatihan, jadi bisa dipakai kalau tidak ada label.
# **Cross-encoder IndoBERT:** kedua teks masuk bersamaan dan dilatih dengan loss MSE. Biasanya lebih akurat tapi
# butuh label.

# %%
with stage("biencoder"):
    if USE_PRETRAINED:
        cos_train = np.sum(e5_embed(train["a"].tolist()) * e5_embed(train["b"].tolist()), axis=1)
        cos_test = np.sum(e5_embed(test["a"].tolist()) * e5_embed(test["b"].tolist()), axis=1)
        record("biencoder_e5", cos_train, cos_test)
    else:
        cos_train, cos_test = x_train[:, 0], x_test[:, 0]

mean_y, std_y = y.mean(), y.std()
oof_cross, hold_cross = np.zeros(len(y)), np.zeros(len(y_holdout))
with stage("cross_encoder"):
    if USE_PRETRAINED:
        for model_name, seed in SETTINGS["model_runs"]:
            for fold in range(SETTINGS["folds"]):
                fit, val = folds != fold, folds == fold
                pred_val, pred_hold = train_predict(
                    model_name, train.loc[fit, "a"].tolist(), (y[fit] - mean_y) / std_y,
                    [(train.loc[val, "a"].tolist(), train.loc[val, "b"].tolist()),
                     (test["a"].tolist(), test["b"].tolist())],
                    problem="mse", num_labels=1, train_pairs=train.loc[fit, "b"].tolist(),
                    epochs=SETTINGS["epochs"], seed=seed + fold, batch_size=16,
                )
                oof_cross[val] += pred_val * std_y + mean_y
                hold_cross += (pred_hold * std_y + mean_y) / SETTINGS["folds"]
        oof_cross /= len(SETTINGS["model_runs"])
        hold_cross /= len(SETTINGS["model_runs"])
        record("cross_encoder", oof_cross, hold_cross)
    else:
        oof_cross, hold_cross = oof_ridge, hold_ridge

# %% [markdown]
# ## 9. Post-processing
#
# Gabungkan bi-encoder dan cross-encoder dengan rata-rata peringkat tanpa bobot. Lalu kalibrasi isotonik supaya
# skala skornya kembali ke 0 sampai 5, yang penting kalau metriknya RMSE. Kalibrasi difit di OOF dan di dalam fold.

# %%
def to_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy()


oof_blend = (to_rank(oof_cross) + to_rank(cos_train)) / 2
hold_blend = (to_rank(hold_cross) + to_rank(cos_test)) / 2
record("gabungan_peringkat", oof_blend, hold_blend)

oof_final = np.zeros(len(y))
for fold in range(SETTINGS["folds"]):
    fit, val = folds != fold, folds == fold
    oof_final[val] = IsotonicRegression(out_of_bounds="clip").fit(oof_blend[fit], y[fit]).predict(oof_blend[val])
hold_final = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y).predict(hold_blend)
record("gabungan_plus_kalibrasi", oof_final, hold_final)

# %% [markdown]
# **Deteksi parafrase:** threshold F1 difit nested dari skor OOF.

# %%
is_paraphrase, is_paraphrase_holdout = y >= PARAPHRASE_MIN_SCORE, y_holdout >= PARAPHRASE_MIN_SCORE


def best_cut(scores, target) -> float:
    grid = np.quantile(scores, np.linspace(0.3, 0.95, 66))
    return float(grid[int(np.argmax([f1_score(target, scores >= c, zero_division=0) for c in grid]))])


nested = np.zeros(len(y), dtype=bool)
for fold in range(SETTINGS["folds"]):
    fit, val = folds != fold, folds == fold
    nested[val] = oof_blend[val] >= best_cut(oof_blend[fit], is_paraphrase[fit])
cut = best_cut(oof_blend, is_paraphrase)
paraphrase_scores = {"f1_oof": round(f1_score(is_paraphrase, nested), 4),
                     "f1_holdout": round(f1_score(is_paraphrase_holdout, hold_blend >= cut), 4)}
print("parafrase:", paraphrase_scores)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Skor menurut sumber data** pada holdout, karena beda sumber bisa beda kesulitan.

# %%
test.assign(pred=hold_final).groupby("data").apply(
    lambda g: pd.Series({"baris": len(g), "pearson": round(pearson(g["score"], g["pred"]), 3)}), include_groups=False
).sort_values("baris", ascending=False)

# %% [markdown]
# **Prediksi dibanding skor asli.**

# %%
fig, ax = plt.subplots()
ax.scatter(y_holdout, hold_final, s=6, alpha=0.4, color="#4C72B0")
ax.set(title="Prediksi dan skor asli pada holdout", xlabel="skor asli", ylabel="prediksi")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Salah paling jauh.**

# %%
errors = test.assign(pred=hold_final.round(2), selisih=np.abs(hold_final - y_holdout).round(2))
errors.sort_values("selisih", ascending=False)[["a", "b", "score", "pred"]].head(6)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": test.index, "skor": hold_final.round(3), "parafrase": (hold_blend >= cut).astype(int)}).to_csv(
    RESULT_DIR / "submission.csv", index=False
)
run_summary("kemiripan", len(train),
            {"parafrase": paraphrase_scores,
             **{r["model"]: {"pearson_oof": r["pearson_oof"], "pearson_holdout": r["pearson_holdout"]} for r in results}},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Gabungan peringkat dengan kalibrasi: Pearson holdout 0,7788, sedangkan cosine TF-IDF 0,6111.
# - Bi-encoder saja kalah dari cross-encoder (0,6659 dibanding 0,7669). Kalau pasangan terlalu banyak, bi-encoder jauh lebih murah (44 detik dibanding 575 detik).
# - Gabungan lebih rendah dari cross-encoder di OOF, tapi lebih tinggi di holdout. Gabungan cenderung lebih tahan saat data bergeser.
# - Sumber terlemah: `MSRpar` 0,582 dan `answer-answer` 0,614. F1 parafrase holdout hanya 0,5745.
