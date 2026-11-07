# %% [markdown]
# # 23 Konsistensi Judul dan Isi
#
# Menilai apakah klaim sesuai dengan isi dokumen. Contoh: kalimat pertama ringkasan IndoSum sebagai judul, dengan
# negatif buatan: artikel lain, dan entitas atau angka yang ditukar.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Memeriksa apakah judul sesuai isi |
# | **Unit prediksi** | Satu pasangan judul dan isi mendapat label biner |
# | **Metrik** | Macro-F1 |
# | **Masalah terbesar** | Negatif dibuat sendiri, jadi skor hanya berlaku untuk tiga jenis kesalahan yang disimulasikan. Perubahan angka paling sulit dideteksi. |
# | **Data contoh** | IndoSum: 4.498 artikel, negatif buatan |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta mendeteksi judul yang tidak sesuai isi, clickbait, atau klaim yang diubah. Kalau
# negatifnya buatan, cari dulu **cara pembuatannya** dari data, karena itu biasanya sumber sinyal terbesar.
#
# **Yang perlu diganti kalau datanya beda:**
# - Blok pembuatan pasangan di bagian 5 tidak dipakai kalau label sudah ada.
# - `make_folds`: artikel yang sama wajib satu fold.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya fitur relasi, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
NEGATIVE_SHARE = 0.5
MAX_LEN = 256
BODY_WORDS = 200

SETTINGS = {
    "cepat": {"folds": 3, "epochs": 2, "model_runs": [("indobert", SEED)]},
    "penuh": {"folds": 5, "epochs": 3, "model_runs": [("indobert", SEED), ("indobert", SEED + 1)]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include clf_common

# %% include transformer_seqcls

# %% include indosum

# %% [markdown]
# ## 4. EDA
#
# Muat artikel. Tiap artikel menghasilkan satu pasangan, jadi jumlah pasangan sama dengan `N_ROWS`.

# %%
articles = load_indosum("test.01.jsonl", N_ROWS)
articles = pd.concat([articles, load_indosum("dev.01.jsonl", N_ROWS)], ignore_index=True).head(N_ROWS)
articles = articles[articles["first_summary"].str.split().str.len() >= 6].reset_index(drop=True)
print(f"artikel: {len(articles)}")
print(articles["category"].value_counts().to_dict())
print("contoh judul:", articles["first_summary"][0][:120])
text_profile(articles["first_summary"])

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Tidak ada label asli | Buat tiga jenis negatif: artikel lain, entitas ditukar, angka diubah |
# | Satu artikel menghasilkan beberapa pasangan | Fold berdasarkan artikel |
# | Isi artikel panjang | Isi dipotong karena batas token model |

# %% [markdown]
# ## 5. Preprocessing
#
# Separuh pasangan benar, separuh negatif dengan tiga jenis yang sulitnya berbeda: artikel lain di kategori yang
# sama, entitas ditukar, dan angka diubah. Isi artikel dipotong, karena token model terbatas.

# %%
ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+\b")


def swap_entity(title: str, pool: list[str], rng) -> str | None:
    found = ENTITY_RE.findall(title)
    if not found:
        return None
    return title.replace(str(rng.choice(found)), str(rng.choice(pool)), 1)


def change_number(title: str, rng) -> str | None:
    found = NUMBER_RE.findall(title)
    if not found:
        return None
    number = str(rng.choice(found))
    return title.replace(number, str(int(number) + int(rng.integers(1, 9))), 1)


def build_pairs() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    entity_pool = sorted(set(ENTITY_RE.findall(" ".join(articles["first_summary"]))))
    rows = []
    for i, row in articles.iterrows():
        title, kind = row["first_summary"], "asli"
        if rng.random() < NEGATIVE_SHARE:
            choice = rng.choice(["artikel_lain", "entitas", "angka"])
            if choice == "entitas":
                changed = swap_entity(title, entity_pool, rng)
            elif choice == "angka":
                changed = change_number(title, rng)
            else:
                others = articles[(articles["category"] == row["category"]) & (articles.index != i)]
                changed = others["first_summary"].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            if changed:
                title, kind = changed, str(choice)
        rows.append({"article_id": i, "title": title, "body": " ".join(row["article"].split()[:BODY_WORDS]),
                     "jenis": kind, "label": int(kind == "asli")})
    return pd.DataFrame(rows)


pairs = build_pairs()
print(pairs["jenis"].value_counts().to_dict())
holdout_ids = np.random.default_rng(SEED).choice(len(articles), int(0.2 * len(articles)), replace=False)
is_holdout = pairs["article_id"].isin(holdout_ids)
train, holdout = pairs[~is_holdout].reset_index(drop=True), pairs[is_holdout].reset_index(drop=True)
y, y_holdout = train["label"].to_numpy(), holdout["label"].to_numpy()
print(f"latih={len(train)} | holdout={len(holdout)}")

# %% [markdown]
# ## 6. Validasi
#
# Fold berdasarkan artikel, jadi artikel yang sama tidak pernah ada di latih dan validasi sekaligus. Metriknya
# macro-F1. Semua fitur dihitung per pasangan dan tidak memakai label.

# %%
folds = make_folds(y, train["article_id"], SETTINGS["folds"])
print("baris per fold:", np.bincount(folds).tolist())

# %% [markdown]
# ## 7. Baseline
#
# Tebak mayoritas, lalu fitur relasi: seberapa banyak kata, entitas, dan angka judul yang muncul di isi. Fitur ini
# sering sangat kuat dan tetap jalan tanpa model dari luar.

# %%
def relation_features(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for title, body in zip(frame["title"], frame["body"]):
        words = set(title.lower().split())
        body_lower = body.lower()
        body_set = set(body_lower.split())
        entities, numbers = ENTITY_RE.findall(title), NUMBER_RE.findall(title)
        rows.append([
            len(words & body_set) / max(len(words), 1),
            np.mean([e in body for e in entities]) if entities else 1.0,
            np.mean([n in body for n in numbers]) if numbers else 1.0,
            float(all(e in body for e in entities)),
            float(all(n in body for n in numbers)),
            len(title.split()),
        ])
    return np.array(rows, dtype=float)


majority = np.bincount(y).argmax()
record("mayoritas", np.full(len(y), majority), np.full(len(y_holdout), majority))
x_train, x_holdout = relation_features(train), relation_features(holdout)
oof_lr, hold_lr = np.zeros(len(y)), np.zeros(len(y_holdout))
for fold in range(SETTINGS["folds"]):
    fit, val = folds != fold, folds == fold
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_train[fit], y[fit])
    oof_lr[val] = model.predict_proba(x_train[val])[:, 1]
    hold_lr += model.predict_proba(x_holdout)[:, 1] / SETTINGS["folds"]
record("fitur_relasi", (oof_lr >= 0.5).astype(int), (hold_lr >= 0.5).astype(int))

# %% [markdown]
# ## 8. Model Utama
#
# Cross-encoder: judul dan isi masuk bersamaan, jadi model bisa membandingkan keduanya.

# %%
oof_cross, hold_cross = oof_lr.copy(), hold_lr.copy()
with stage("cross_encoder"):
    if USE_PRETRAINED:
        oof_cross, hold_cross = np.zeros(len(y)), np.zeros(len(y_holdout))
        for model_name, seed in SETTINGS["model_runs"]:
            for fold in range(SETTINGS["folds"]):
                fit, val = folds != fold, folds == fold
                pred_val, pred_hold = train_predict(
                    model_name, train.loc[fit, "title"].tolist(), y[fit],
                    [(train.loc[val, "title"].tolist(), train.loc[val, "body"].tolist()),
                     (holdout["title"].tolist(), holdout["body"].tolist())],
                    num_labels=2, train_pairs=train.loc[fit, "body"].tolist(), epochs=SETTINGS["epochs"],
                    seed=seed + fold, batch_size=16, max_len=MAX_LEN,
                )
                oof_cross[val] += pred_val[:, 1] / len(SETTINGS["model_runs"])
                hold_cross += pred_hold[:, 1] / (SETTINGS["folds"] * len(SETTINGS["model_runs"]))
        record("cross_encoder", (oof_cross >= 0.5).astype(int), (hold_cross >= 0.5).astype(int))

# %% [markdown]
# ## 9. Post-processing
#
# Stacking cross-encoder dengan fitur relasi, karena keduanya menangkap kesalahan berbeda. Stacker dan threshold
# difit di dalam fold, jadi skornya jujur.

# %%
def best_threshold(scores, target) -> float:
    grid = np.arange(0.2, 0.81, 0.01)
    return float(grid[int(np.argmax([macro_f1(target, scores >= t) for t in grid]))])


x_stack = np.column_stack([oof_cross, x_train])
x_stack_hold = np.column_stack([hold_cross, x_holdout])
oof_stack, hold_stack = np.zeros(len(y)), np.zeros(len(y_holdout))
oof_pred = np.zeros(len(y), dtype=int)
for fold in range(SETTINGS["folds"]):
    fit, val = folds != fold, folds == fold
    stacker = LogisticRegression(max_iter=2000).fit(x_stack[fit], y[fit])
    oof_stack[val] = stacker.predict_proba(x_stack[val])[:, 1]
    oof_pred[val] = oof_stack[val] >= best_threshold(stacker.predict_proba(x_stack[fit])[:, 1], y[fit])
    hold_stack += stacker.predict_proba(x_stack_hold)[:, 1] / SETTINGS["folds"]
hold_pred = (hold_stack >= best_threshold(oof_stack, y)).astype(int)
record("stacking_plus_threshold", oof_pred, hold_pred)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Skor menurut jenis pasangan.** Ini yang paling penting: jenis mana yang berhasil dan gagal dideteksi.

# %%
holdout.assign(pred=hold_pred, benar=hold_pred == y_holdout).groupby("jenis")["benar"].agg(["size", "mean"]).round(3)

# %% [markdown]
# **Salah prediksi.**

# %%
errors = holdout.assign(pred=hold_pred)
errors[errors["label"] != errors["pred"]][["title", "jenis", "label"]].head(8)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": holdout.index, "label": hold_pred}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("konsistensi", len(pairs), {r["model"]: {"oof": r["oof"], "holdout": r["holdout"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Stacking terbaik: holdout 0,9533, sedangkan fitur relasi 0,919 dan cross-encoder 0,9127.
# - Fitur sederhana sudah mengalahkan cross-encoder. Keduanya saling melengkapi.
# - Akurasi per jenis: artikel lain 1,0, entitas 0,965, asli 0,947, angka 0,886.
# - Skor pada keluhan MBG asli bisa jauh lebih rendah, karena kesalahan aslinya lebih halus.
