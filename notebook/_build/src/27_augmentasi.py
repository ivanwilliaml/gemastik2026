# %% [markdown]
# # 27 Parafrase dan Augmentasi
#
# Menambah data latih dengan teks baru, lalu diukur apakah benar-benar menaikkan skor. Contoh: 500 baris SmSA berlabel.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Menambah data dengan parafrase dan augmentasi |
# | **Unit prediksi** | Satu teks latih mendapat salinan yang diubah |
# | **Metrik** | Macro-F1, dan kemiripan makna antara teks asli dan salinan |
# | **Masalah terbesar** | Holdout bergeser dan label sedikit (500 baris): OOF tanpa augmentasi 0,8336, tapi holdout hanya 0,6265. Augmentasi yang salah tempat juga membuat skor palsu. |
# | **Data contoh** | SmSA: 500 latih, 500 holdout |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** data berlabel sedikit, atau soal meminta membuat variasi teks. Kalau ada banyak data tanpa label,
# coba folder `08` dulu, karena pseudo-label memakai teks asli.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `CLASSES`.
# - Jenis augmentasi di `AUGMENTERS`. Hapus yang label-nya sering berubah.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya augmentasi kata, tanpa file model dari luar
SEED = 42
N_LABELED = 500
CLASSES = ["negative", "neutral", "positive"]

SETTINGS = {
    "cepat": {"folds": 3, "epochs": 4, "seeds": [SEED]},
    "penuh": {"folds": 5, "epochs": 6, "seeds": [SEED, SEED + 1]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include clf_common

# %% include transformer_seqcls

# %% include embedding

# %% include seq2seq

# %% [markdown]
# ## 4. EDA
#
# Muat data. Test resmi dipakai sebagai holdout.

# %%
def read_smsa(split: str) -> pd.DataFrame:
    path = EXTRA_DATA_DIR / "indonlu/dataset/smsa_doc-sentiment-prosa" / f"{split}_preprocess.tsv"
    return pd.read_csv(path, sep="\t", header=None, names=["text", "label"])


pool = read_smsa("train")
train = pool.groupby("label", group_keys=False).sample(frac=N_LABELED / len(pool), random_state=SEED).reset_index(drop=True)
holdout = read_smsa("test")
print(f"latih={len(train)} | holdout={len(holdout)}")
print("proporsi kelas:", train["label"].value_counts(normalize=True).round(3).to_dict())
text_profile(train["text"])

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Label hanya 500 baris | Augmentasi hanya di data latih dalam fold |
# | Netral 10,4% | Semua varian memakai fold yang sama |
# | Back translation bisa mengubah makna | Kemiripan diukur (median 0,934), dan contoh dibaca manual |

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan. **Aturan utama: augmentasi hanya untuk data latih di dalam fold.** Kalau dilakukan sebelum
# pembagian fold, salinan teks bocor ke validasi dan skornya palsu.

# %%
class_to_id = {name: i for i, name in enumerate(CLASSES)}
train["clean"] = train["text"].map(clean_text)
holdout["clean"] = holdout["text"].map(clean_text)
train["group"] = near_duplicate_groups(train["clean"].tolist())
y = train["label"].map(class_to_id).to_numpy()
y_holdout = holdout["label"].map(class_to_id).to_numpy()

# %% [markdown]
# **Augmentasi tanpa model.** Hapus kata acak, tukar kata bertetangga, dan ganti dengan bentuk informal dari kamus.

# %%
slang = load_slang_dict()
formal_to_slang: dict[str, str] = {}
for informal, formal in slang.items():
    if " " not in formal and len(informal) < len(formal):
        formal_to_slang.setdefault(formal, informal)


def word_dropout(text: str, rng, share: float = 0.1) -> str:
    words = text.split()
    kept = [w for w in words if rng.random() > share]
    return " ".join(kept) if len(kept) >= 3 else text


def word_swap(text: str, rng, n: int = 2) -> str:
    words = text.split()
    for _ in range(n):
        if len(words) > 2:
            i = rng.integers(0, len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def to_informal(text: str, rng, share: float = 0.3) -> str:
    return " ".join(formal_to_slang[w] if w in formal_to_slang and rng.random() < share else w for w in text.split())


# %% [markdown]
# **Back-translation.** Indonesia ke Inggris lalu kembali. Hasilnya dihitung sekali untuk semua teks latih, tapi tetap
# hanya masuk ke fold latih.

# %%
back_translated = None
if USE_PRETRAINED:
    with stage("back_translation"):
        tok_forward, model_forward = load_seq2seq("marian_id_en")
        english = generate_texts(model_forward, tok_forward, train["clean"].tolist(), max_input=128, max_new=128, beams=1)
        del model_forward
        tok_back, model_back = load_seq2seq("marian_en_id")
        back_translated = generate_texts(model_back, tok_back, english, max_input=128, max_new=128, beams=1)
        del model_back
        torch.cuda.empty_cache()
    print("asli:", train["clean"][0][:90])
    print("hasil:", back_translated[0][:90])

# %% [markdown]
# ## 6. Validasi
#
# Skema `StratifiedGroupKFold`, metrik macro-F1. Semua varian memakai fold yang sama, supaya perbandingannya adil.
# Augmentasi diterima kalau interval perbedaannya jelas di atas nol, bukan hanya rata-ratanya naik.

# %%
folds = make_folds(y, train["group"], SETTINGS["folds"])
print("baris per fold:", np.bincount(folds).tolist())


def paired_gain(base_pred: np.ndarray, new_pred: np.ndarray, n_boot: int = 500) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(y), len(y))
        gains.append(macro_f1(y[pick], new_pred[pick]) - macro_f1(y[pick], base_pred[pick]))
    return {"kenaikan": round(float(np.mean(gains)), 4), "p_lebih_baik": round(float(np.mean(np.array(gains) > 0)), 3)}


# %% [markdown]
# ## 7. Baseline
#
# Tanpa augmentasi. Kalau `USE_PRETRAINED=False`, classifier memakai TF-IDF + regresi logistik.

# %%
def fit_predict(texts, labels, predict_sets, seed):
    if USE_PRETRAINED:
        weights = len(labels) / (len(CLASSES) * np.bincount(labels, minlength=len(CLASSES)))
        return train_predict("indobertweet", list(texts), labels, predict_sets, num_labels=len(CLASSES),
                             class_weights=weights, epochs=SETTINGS["epochs"], seed=seed)
    vector = _Tfidf(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    model = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced").fit(vector.fit_transform(texts), labels)
    return [model.predict_proba(vector.transform(s)) for s in predict_sets]


def run_cv(augment_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validation. augment_fn menerima indeks latih dan mengembalikan teks tambahan."""
    oof = np.zeros((len(y), len(CLASSES)))
    hold = np.zeros((len(y_holdout), len(CLASSES)))
    n_runs = SETTINGS["folds"] * len(SETTINGS["seeds"])
    for seed in SETTINGS["seeds"]:
        for fold in range(SETTINGS["folds"]):
            fit_idx, val_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
            texts = train["clean"].iloc[fit_idx].tolist()
            labels = y[fit_idx]
            if augment_fn is not None:
                texts = texts + augment_fn(fit_idx)
                labels = np.concatenate([labels, y[fit_idx]])
            pred_val, pred_hold = fit_predict(texts, labels, [train["clean"].iloc[val_idx].tolist(), holdout["clean"].tolist()], seed + fold)
            oof[val_idx] += pred_val / len(SETTINGS["seeds"])
            hold += pred_hold / n_runs
    return oof, hold


with stage("tanpa_augmentasi"):
    oof_base, hold_base = run_cv()
record("tanpa_augmentasi", oof_base.argmax(1), hold_base.argmax(1))

# %% [markdown]
# ## 8. Model Utama
#
# Latih ulang dengan tiap augmentasi pada fold yang sama.

# %%
def word_augment(fn):
    def augment(index: np.ndarray) -> list[str]:
        rng = np.random.default_rng(SEED)
        return [fn(train["clean"].iloc[i], rng) for i in index]

    return augment


AUGMENTERS = {"hapus_kata": word_augment(word_dropout), "tukar_kata": word_augment(word_swap), "slang": word_augment(to_informal)}
if back_translated is not None:
    AUGMENTERS["back_translation"] = lambda index: [back_translated[i] for i in index]

outcomes = {"tanpa_augmentasi": (oof_base, hold_base)}
for name, fn in AUGMENTERS.items():
    with stage(name):
        outcomes[name] = run_cv(fn)
    record(name, outcomes[name][0].argmax(1), outcomes[name][1].argmax(1))

# %% [markdown]
# ## 9. Post-processing
#
# Pilih varian dengan peluang kenaikan terbaik di OOF. Kalau tidak ada yang melewati 0,9, tetap tanpa augmentasi.

# %%
gains = {n: paired_gain(oof_base.argmax(1), o[0].argmax(1)) for n, o in outcomes.items() if n != "tanpa_augmentasi"}
gain_table = pd.DataFrame(gains).T.sort_values("p_lebih_baik", ascending=False)
best_name = gain_table.index[0] if gain_table["p_lebih_baik"].iloc[0] >= 0.9 else "tanpa_augmentasi"
print("dipilih:", best_name)
gain_table

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Apakah makna dan label terjaga?** Kemiripan embedding teks asli dan hasil augmentasi. Kalau terlalu tinggi, teks hampir
# sama dan tidak menambah informasi. Kalau terlalu rendah, label mungkin sudah berubah.

# %%
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(train), min(200, len(train)), replace=False)
originals = train["clean"].iloc[sample_idx].tolist()
variants = {"hapus_kata": [word_dropout(t, rng) for t in originals], "tukar_kata": [word_swap(t, rng) for t in originals],
            "slang": [to_informal(t, rng) for t in originals]}
if back_translated is not None:
    variants["back_translation"] = [back_translated[i] for i in sample_idx]
base_vectors = embed_texts(originals)
similarity_rows = []
for name, texts in variants.items():
    changed = np.mean([a != b for a, b in zip(originals, texts)])
    similarity_rows.append({"augmentasi": name, "kemiripan_median": round(float(np.median(np.sum(base_vectors * embed_texts(texts), axis=1))), 3), "teks_berubah": round(float(changed), 3)})
pd.DataFrame(similarity_rows)

# %% [markdown]
# **Contoh hasil,** untuk dicek manual apakah label masih benar.

# %%
pd.DataFrame({"asli": [t[:70] for t in originals[:5]], **{k: [t[:70] for t in v[:5]] for k, v in variants.items()}})

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
final_hold = outcomes[best_name][1].argmax(1)
pd.DataFrame({"id": holdout.index, "label": [CLASSES[i] for i in final_hold]}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("augmentasi", len(train), {"dipilih": best_name, **{r["model"]: {"oof": r["oof"], "holdout": r["holdout"]} for r in results}},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Tidak ada augmentasi yang lolos syarat peluang 0,9. Terbaik: slang, kenaikan OOF 0,0173 dengan peluang 0,884.
# - Semua augmentasi menaikkan holdout dari 0,6265 ke 0,76-0,80. Augmentasi juga menggandakan jumlah langkah latih, jadi kenaikan ini belum tentu karena isi augmentasinya. Perlu pembanding epoch lebih banyak tanpa augmentasi.
# - Aturan memilih dari OOF menolak semua augmentasi, padahal holdout-nya naik. Saat data sangat sedikit dan bergeser, OOF bisa kurang peka.
# - Back translation paling mahal, dan holdout-nya terendah di antara augmentasi (0,7577).
