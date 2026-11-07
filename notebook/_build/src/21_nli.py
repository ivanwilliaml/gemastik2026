# %% [markdown]
# # 21 Entailment atau NLI
#
# Menilai apakah teks kedua didukung oleh teks pertama. Contoh: dataset WReTE, sangat kecil dengan 300 pasangan latih.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Menilai apakah satu teks mendukung teks lain |
# | **Unit prediksi** | Satu pasangan teks mendapat label biner |
# | **Metrik** | Macro-F1 |
# | **Masalah terbesar** | Data sangat kecil (300 latih, 100 test), jadi interval lebar. Kedua kalimat juga sangat mirip, dan perbedaan kecil seperti negasi menentukan label. |
# | **Data contoh** | WReTE: 300 latih, 50 valid, 100 test |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** data berisi pasangan teks dengan label mendukung atau tidak. Termasuk cek klaim terhadap sumber,
# misalnya klaim MBG terhadap berita. Kalau targetnya skor, pakai folder `20`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `read_split`, kolom `sent_A`, `sent_B`, `label`.
# - `ENTAIL_LABEL`, nama label positif.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya fitur kata, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
ENTAIL_LABEL = "Entail_or_Paraphrase"

SETTINGS = {
    "cepat": {"epochs": 5, "seeds": [SEED]},
    "penuh": {"epochs": 8, "seeds": [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include transformer_seqcls

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat data. Datanya berasal dari revisi Wikipedia, jadi banyak pasangan yang hampir sama persis.

# %%
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from transformers import AutoModelForSequenceClassification as NliModel  # noqa: E402
from transformers import AutoTokenizer as NliTokenizer  # noqa: E402


def read_split(split: str) -> pd.DataFrame:
    path = EXTRA_DATA_DIR / "indonlu/dataset/wrete_entailment-ui" / f"{split}_preprocess.csv"
    return pd.read_csv(path, encoding_errors="ignore").head(N_ROWS)


train, valid, test = read_split("train"), read_split("valid"), read_split("test")
print(f"latih={len(train)} | valid={len(valid)} | test={len(test)}")
print("proporsi positif:", round((train["label"] == ENTAIL_LABEL).mean(), 3))
train[["sent_A", "sent_B", "label"]].head(3)

# %% [markdown]
# Seberapa berbeda dua kalimat dalam satu pasangan, per kelas.

# %%
def word_overlap(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


train["overlap"] = [word_overlap(a, b) for a, b in zip(train["sent_A"], train["sent_B"])]
fig, ax = plt.subplots()
for label, part in train.groupby("label"):
    part["overlap"].plot.hist(bins=20, alpha=0.6, ax=ax, label=label)
ax.set(title="Kemiripan kata dalam pasangan", xlabel="Jaccard", ylabel="jumlah pasangan")
ax.legend()
plt.tight_layout()
plt.show()
print(train.groupby("label")["overlap"].median().round(3).to_dict())

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Test hanya 100 baris | Valid digabung ke latih untuk CV, jangan percaya selisih kecil |
# | Kalimat hampir sama | Negasi dan angka tidak diubah |
# | Proporsi positif 61% | Macro-F1 dipakai dan threshold difit di OOF |

# %% [markdown]
# ## 5. Preprocessing
#
# Pembersihan ringan, dan negasi serta angka tidak diubah karena sangat menentukan. Label dijadikan 0 dan 1.

# %%
for frame in (train, valid, test):
    frame["a"], frame["b"] = frame["sent_A"].map(clean_text), frame["sent_B"].map(clean_text)
    frame["y"] = (frame["label"] == ENTAIL_LABEL).astype(int)
y_train, y_valid, y_test = train["y"].to_numpy(), valid["y"].to_numpy(), test["y"].to_numpy()

# %% [markdown]
# ## 6. Validasi
#
# Valid dan test sangat kecil, jadi interval noise-nya lebar. Karena itu valid digabung ke latih untuk
# cross-validation, dan test hanya untuk laporan. Metriknya macro-F1.

# %%
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


assert abs(macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) - (2 / 3 + 0.8) / 2) < 1e-9
pool = pd.concat([train, valid], ignore_index=True)
y = pool["y"].to_numpy()
folds = np.zeros(len(pool), dtype=int)
for fold, (_, val_idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(pool, y)):
    folds[val_idx] = fold
results: list[dict] = []


def bootstrap_width(y_true, y_pred, n_boot: int = 300) -> str:
    rng = np.random.default_rng(SEED)
    scores = [macro_f1(y_true[p], y_pred[p]) for p in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))]
    return f"{np.percentile(scores, 2.5):.3f}-{np.percentile(scores, 97.5):.3f}"


def record(name: str, oof, holdout) -> None:
    results.append({"model": name, "oof": round(macro_f1(y, oof), 4), "oof_ci95": bootstrap_width(y, oof),
                    "test": round(macro_f1(y_test, holdout), 4)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Tebak mayoritas, lalu regresi logistik dari fitur perbedaan: kemiripan kata, selisih panjang, kata baru, angka,
# dan negasi.

# %%
NEGATIONS = {"tidak", "bukan", "belum", "tanpa", "jangan", "tak"}


def pair_features(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for a, b in zip(frame["a"], frame["b"]):
        set_a, set_b = set(a.split()), set(b.split())
        rows.append([word_overlap(a, b), len(set_b - set_a) / max(len(set_b), 1), len(set_a - set_b) / max(len(set_a), 1),
                     len(set_b) / max(len(set_a), 1), len(NEGATIONS & set_a) != len(NEGATIONS & set_b),
                     set(re.findall(r"\d+", a)) != set(re.findall(r"\d+", b))])
    return np.array(rows, dtype=float)


majority = np.bincount(y).argmax()
record("mayoritas", np.full(len(y), majority), np.full(len(y_test), majority))
x_pool, x_test = pair_features(pool), pair_features(test)
oof_lr, hold_lr = np.zeros(len(y)), np.zeros(len(y_test))
for fold in range(5):
    fit, val = folds != fold, folds == fold
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_pool[fit], y[fit])
    oof_lr[val] = model.predict_proba(x_pool[val])[:, 1]
    hold_lr += model.predict_proba(x_test)[:, 1] / 5
record("fitur_perbedaan", (oof_lr >= 0.5).astype(int), (hold_lr >= 0.5).astype(int))

# %% [markdown]
# ## 8. Model Utama
#
# **NLI pretrained (mDeBERTa):** sudah dilatih untuk entailment, jadi tidak butuh latihan, cocok untuk data kecil
# seperti ini. **Cross-encoder IndoBERT:** dilatih dari nol pada data ini, dengan beberapa seed karena datanya kecil.

# %%
@torch.inference_mode()
def pretrained_entailment(frame: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
    tokenizer = NliTokenizer.from_pretrained(MODEL_DIR / "mdeberta_nli")
    model = NliModel.from_pretrained(MODEL_DIR / "mdeberta_nli").to(DEVICE).eval()
    entail = model.config.label2id.get("entailment", 0)
    chunks = []
    for start in range(0, len(frame), batch_size):
        tokens = tokenizer(frame["a"].iloc[start : start + batch_size].tolist(), frame["b"].iloc[start : start + batch_size].tolist(),
                           padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        chunks.append(torch.softmax(model(**tokens).logits.float(), 1)[:, entail].cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(chunks).numpy()


oof_nli, hold_nli = oof_lr, hold_lr
oof_cross, hold_cross = oof_lr, hold_lr
if USE_PRETRAINED:
    with stage("nli_pretrained"):
        oof_nli, hold_nli = pretrained_entailment(pool), pretrained_entailment(test)
    record("nli_pretrained_0_5", (oof_nli >= 0.5).astype(int), (hold_nli >= 0.5).astype(int))
    with stage("cross_encoder"):
        oof_cross, hold_cross = np.zeros(len(y)), np.zeros(len(y_test))
        for seed in SETTINGS["seeds"]:
            for fold in range(5):
                fit, val = folds != fold, folds == fold
                pred_val, pred_hold = train_predict(
                    "indobert", pool.loc[fit, "a"].tolist(), y[fit],
                    [(pool.loc[val, "a"].tolist(), pool.loc[val, "b"].tolist()), (test["a"].tolist(), test["b"].tolist())],
                    num_labels=2, train_pairs=pool.loc[fit, "b"].tolist(), epochs=SETTINGS["epochs"],
                    seed=seed + fold, batch_size=16,
                )
                oof_cross[val] += pred_val[:, 1] / len(SETTINGS["seeds"])
                hold_cross += pred_hold[:, 1] / (5 * len(SETTINGS["seeds"]))
    record("cross_encoder_0_5", (oof_cross >= 0.5).astype(int), (hold_cross >= 0.5).astype(int))

# %% [markdown]
# ## 9. Post-processing
#
# Gabungkan tiga sumber dengan regresi logistik, sebagai stacking di dalam fold. Threshold ikut difit dari fold
# lain, jadi skornya tetap jujur.

# %%
def best_threshold(scores, target) -> float:
    grid = np.arange(0.2, 0.81, 0.02)
    return float(grid[int(np.argmax([macro_f1(target, scores >= t) for t in grid]))])


stack_pool = np.column_stack([oof_lr, oof_nli, oof_cross])
stack_test = np.column_stack([hold_lr, hold_nli, hold_cross])
oof_stack, hold_stack = np.zeros(len(y)), np.zeros(len(y_test))
oof_pred = np.zeros(len(y), dtype=int)
for fold in range(5):
    fit, val = folds != fold, folds == fold
    stacker = LogisticRegression(C=1.0, max_iter=2000).fit(stack_pool[fit], y[fit])
    fit_scores = stacker.predict_proba(stack_pool[fit])[:, 1]
    oof_stack[val] = stacker.predict_proba(stack_pool[val])[:, 1]
    oof_pred[val] = oof_stack[val] >= best_threshold(fit_scores, y[fit])
    hold_stack += stacker.predict_proba(stack_test)[:, 1] / 5
hold_pred = (hold_stack >= best_threshold(oof_stack, y)).astype(int)
record("stacking_plus_threshold", oof_pred, hold_pred)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.** Perhatikan lebarnya interval, karena data kecil membuat selisih kecil tidak berarti.

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Confusion matrix OOF.**

# %%
plot_confusion(y, oof_pred, ["tidak_mendukung", "mendukung"], "Confusion matrix OOF")

# %% [markdown]
# **Salah prediksi.** Biasanya pasangan yang katanya hampir sama tapi maknanya berubah.

# %%
errors = pool.assign(pred=oof_pred)
errors[errors["y"] != errors["pred"]][["a", "b", "y", "pred"]].head(6)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": test.index, "label": hold_pred}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("nli", len(pool), {r["model"]: {"oof": r["oof"], "test": r["test"]} for r in results}, needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Model NLI pretrained tanpa latihan sudah 0,8827 di test, dan melatih cross-encoder dari 300 baris hanya 0,7868.
# - Stacking test 0,9132, tapi OOF-nya 0,8748 dan lebih rendah dari pretrained 0,8791. Dengan 100 baris test, selisih ini belum bisa disebut nyata.
# - **Kalau label sedikit, coba model pretrained dulu** sebelum melatih sendiri.
