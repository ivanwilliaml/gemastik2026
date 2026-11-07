# %% [markdown]
# # 25 Terjemahan
#
# Menerjemahkan bahasa Indonesia ke Inggris. Contoh: korpus paralel ALT.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Menerjemahkan Indonesia ke Inggris |
# | **Unit prediksi** | Satu kalimat sumber mendapat satu kalimat target |
# | **Metrik** | BLEU dan chrF |
# | **Masalah terbesar** | Data latih 5.000 pasangan terlalu sedikit untuk melatih dari nol, jadi kualitas sangat bergantung pada model pretrained. |
# | **Data contoh** | ALT: 5.000 latih, 500 valid, 1.018 test, kalimat berita |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** data berupa pasangan teks dua bahasa, atau butuh terjemahan untuk memakai model dan kamus berbahasa
# Inggris. Metriknya biasanya BLEU atau chrF.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_split` dan kolom bahasa.
# - `MODEL_NAME`: `marian_id_en` atau `marian_en_id`.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya kamus kata dari data latih, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
MODEL_NAME = "marian_id_en"
SOURCE_LANG, TARGET_LANG = "id", "en"

SETTINGS = {"cepat": {"finetune_rows": 1000, "epochs": 1, "n_test": 500},
            "penuh": {"finetune_rows": 5000, "epochs": 3, "n_test": 1019}}[MODE]

seed_everything(SEED)
install_offline("sacrebleu")
print(SETTINGS)

# %% include text_utils

# %% include seq2seq

# %% [markdown]
# ## 4. EDA
#
# Muat pasangan kalimat. Kolom `translation` berisi kamus per bahasa.

# %%
import sacrebleu  # noqa: E402


def load_split(split: str, n_rows: int) -> pd.DataFrame:
    frame = pd.read_parquet(EXTRA_DATA_DIR / "alt" / f"{split}.parquet")
    output = pd.DataFrame(
        {"source": frame["translation"].map(lambda d: d[SOURCE_LANG]), "target": frame["translation"].map(lambda d: d[TARGET_LANG])}
    )
    output = output[(output["source"].str.len() > 0) & (output["target"].str.len() > 0)]
    return output.head(n_rows).reset_index(drop=True)


train = load_split("train", N_ROWS)
valid = load_split("validation", 500)
test = load_split("test", SETTINGS["n_test"])
print(f"latih={len(train)} | valid={len(valid)} | test={len(test)}")
print("sumber:", train["source"][0][:100])
print("target:", train["target"][0][:100])
pd.DataFrame({"sumber": text_profile(train["source"]), "target": text_profile(train["target"])})

# %% [markdown]
# Rasio panjang sumber dan target. Pasangan yang rasionya ekstrem biasanya salah sejajar.

# %%
ratio = train["source"].str.split().str.len() / train["target"].str.split().str.len().clip(lower=1)
fig, ax = plt.subplots()
ratio.clip(upper=3).plot.hist(bins=40, ax=ax, color="#4C72B0")
ax.set(title="Rasio panjang sumber dan target", xlabel="rasio jumlah kata", ylabel="jumlah pasangan")
plt.tight_layout()
plt.show()
print(f"rasio di luar 0,4-2,5: {((ratio < 0.4) | (ratio > 2.5)).mean():.1%}")

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Tidak ada pasangan dengan rasio panjang ekstrem | Filter tidak membuang apa pun, tapi tetap disiapkan untuk data lain |
# | Huruf besar dan tanda baca memengaruhi skor | Teks tidak diubah |
# | Kalimat median 19 kata | Batas 128 token cukup |

# %% [markdown]
# ## 5. Preprocessing
#
# Huruf besar dan tanda baca dipertahankan, karena memengaruhi skor. Pasangan yang panjangnya sangat tidak wajar
# dibuang hanya dari data latih, sementara data test tetap utuh.

# %%
train = train[(ratio >= 0.4) & (ratio <= 2.5)].reset_index(drop=True)
print(f"latih setelah dibersihkan: {len(train)}")

# %% [markdown]
# ## 6. Validasi
#
# Memakai pembagian resmi. Valid menentukan apakah fine-tune diterima, dan test hanya untuk laporan. Metrik: BLEU, yang
# peka pada urutan kata, dan chrF, yang lebih adil untuk bentuk kata.

# %%
def translation_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    return {"bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2)}


assert translation_scores(["the cat sat on the mat today"], ["the cat sat on the mat today"])["bleu"] > 99
results: list[dict] = []


def record(name: str, pred_valid: list[str], pred_test: list[str]) -> None:
    valid_scores, test_scores = translation_scores(pred_valid, valid["target"].tolist()), translation_scores(pred_test, test["target"].tolist())
    results.append({"metode": name, "bleu_valid": valid_scores["bleu"], "bleu_test": test_scores["bleu"], "chrf_test": test_scores["chrf"]})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Salin teks sumber, lalu kamus kata dari kemunculan bersama di data latih. Tidak butuh model dari luar.

# %%
from collections import Counter, defaultdict  # noqa: E402

record("salin_sumber", valid["source"].tolist(), test["source"].tolist())

co_counts: dict[str, Counter] = defaultdict(Counter)
target_counts: Counter = Counter()
for source, target in zip(train["source"], train["target"]):
    target_words = set(target.lower().split())
    target_counts.update(target_words)
    for word in set(source.lower().split()):
        co_counts[word].update(target_words)
lexicon = {w: max(c, key=lambda t: c[t] / (target_counts[t] + 5)) for w, c in co_counts.items()}


def lexicon_translate(text: str) -> str:
    return " ".join(lexicon.get(word.lower(), word) for word in text.split())


record("kamus_kata", valid["source"].map(lexicon_translate).tolist(), test["source"].map(lexicon_translate).tolist())

# %% [markdown]
# ## 8. Model Utama
#
# MarianMT pretrained. Lalu fine-tune dari data latih, dan hasilnya hanya dipakai kalau skor valid naik.

# %%
valid_final, test_final = valid["source"].map(lexicon_translate).tolist(), test["source"].map(lexicon_translate).tolist()
if USE_PRETRAINED:
    tokenizer, model = load_seq2seq(MODEL_NAME)
    with stage("marian_pretrained"):
        valid_base = generate_texts(model, tokenizer, valid["source"].tolist(), max_input=128, max_new=128)
        test_base = generate_texts(model, tokenizer, test["source"].tolist(), max_input=128, max_new=128)
    record("marian_pretrained", valid_base, test_base)
    with stage("marian_finetune"):
        subset = train.head(SETTINGS["finetune_rows"])
        model = finetune_seq2seq(model, tokenizer, subset["source"].tolist(), subset["target"].tolist(), SETTINGS["epochs"], SEED,
                                 batch_size=32, max_input=128, max_target=128)
        valid_tuned = generate_texts(model, tokenizer, valid["source"].tolist(), max_input=128, max_new=128)
        test_tuned = generate_texts(model, tokenizer, test["source"].tolist(), max_input=128, max_new=128)
    record("marian_finetune", valid_tuned, test_tuned)
    del model
    torch.cuda.empty_cache()
    tuned_better = translation_scores(valid_tuned, valid["target"].tolist())["bleu"] > translation_scores(valid_base, valid["target"].tolist())["bleu"]
    valid_final, test_final = (valid_tuned, test_tuned) if tuned_better else (valid_base, test_base)
    print("dipilih dari valid:", "fine-tune" if tuned_better else "pretrained")

# %% [markdown]
# ## 9. Post-processing
#
# Rapikan keluaran: kata berulang berurutan dibuang, dan spasi sebelum tanda baca dihapus. Kesalahan generator seperti
# ini sering muncul pada kalimat panjang.

# %%
def tidy_translation(text: str) -> str:
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()


record("final_dirapikan", [tidy_translation(t) for t in valid_final], [tidy_translation(t) for t in test_final])
final_test = [tidy_translation(t) for t in test_final]

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Skor menurut panjang kalimat.** Kalimat panjang biasanya lebih lemah.

# %%
buckets = pd.cut(test["source"].str.split().str.len(), [0, 10, 20, 40, 1000], labels=["<=10", "11-20", "21-40", ">40"])
frame = pd.DataFrame({"panjang": buckets, "pred": final_test, "ref": test["target"]})
frame.groupby("panjang", observed=True).apply(
    lambda g: pd.Series({"baris": len(g), "bleu": round(sacrebleu.corpus_bleu(g["pred"].tolist(), [g["ref"].tolist()]).score, 1)}),
    include_groups=False,
)

# %% [markdown]
# **Contoh hasil.**

# %%
pd.DataFrame({"sumber": test["source"].str[:70], "asli": test["target"].str[:70], "hasil": [t[:70] for t in final_test]}).head(5)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": test.index, "terjemahan": final_test}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("terjemahan", len(train), {r["metode"]: {"bleu_test": r["bleu_test"], "chrf_test": r["chrf_test"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Fine-tune hanya sedikit menaikkan BLEU test: 38,44 ke 39,53, setelah 334 detik.
# - Kamus kata buatan sendiri hanya 5,72, jadi tanpa pretrained task ini tidak realistis.
# - BLEU sekitar 39 di semua panjang kalimat.
# - Merapikan keluaran tidak mengubah skor pada data ini.
