# %% [markdown]
# # 26 Normalisasi Teks Informal
#
# Mengubah teks informal menjadi bahasa baku. Contoh: dataset STIF-Indonesia, dari kicauan layanan pelanggan.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta menormalkan slang, atau sebagai langkah awal untuk kata kunci dan topik. Untuk klasifikasi
# dengan IndoBERTweet biasanya tidak perlu, karena model itu sudah terbiasa dengan teks informal.
#
# **Yang perlu diganti kalau datanya beda:**
# - `read_pair`, kalau pasangan teks disimpan dengan format lain.
# - Kamus di `load_slang_dict`, tambahkan istilah domain, misalnya `sppg` dan `bgn` yang tidak perlu diubah.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya kamus, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
PREFIX = "normalisasi: "

SETTINGS = {"cepat": {"epochs": 2}, "penuh": {"epochs": 8}}[MODE]

seed_everything(SEED)
install_offline("sacrebleu", "jiwer")
print(SETTINGS)

# %% include text_utils

# %% include seq2seq

# %% [markdown]
# ## 4. EDA
#
# Muat pasangan informal dan baku. File `.inf` berisi teks informal, dan `.for` berisi teks baku.

# %%
import jiwer  # noqa: E402
import sacrebleu  # noqa: E402


def read_pair(split: str) -> pd.DataFrame:
    folder = EXTRA_DATA_DIR / "stif/data/labelled"
    informal = (folder / f"{split}.inf").read_text(encoding="utf-8").splitlines()
    formal = (folder / f"{split}.for").read_text(encoding="utf-8").splitlines()
    return pd.DataFrame({"informal": informal, "formal": formal}).head(N_ROWS)


train, valid, test = read_pair("train"), read_pair("dev"), read_pair("test")
print(f"latih={len(train)} | valid={len(valid)} | test={len(test)}")
print("informal:", train["informal"][0][:100])
print("baku:", train["formal"][0][:100])
same = (train["informal"] == train["formal"]).mean()
print(f"pasangan identik: {same:.1%}")
text_profile(train["informal"])

# %% [markdown]
# Kata yang paling sering berubah, dan seberapa banyak kata yang berubah per kalimat.

# %%
from collections import Counter  # noqa: E402

changes: Counter = Counter()
for informal, formal in zip(train["informal"], train["formal"]):
    a, b = informal.split(), formal.split()
    if len(a) == len(b):
        changes.update((x, y) for x, y in zip(a, b) if x != y)
print("perubahan tersering:", changes.most_common(12))
print("CER jika teks tidak diubah:", round(jiwer.cer(train["formal"].tolist(), train["informal"].tolist()), 4))

# %% [markdown]
# **Temuan dan keputusan**
#
# ISI_SETELAH_EDA

# %% [markdown]
# ## 5. Preprocessing
#
# Token seperti `xxxuserxxx` dan `xxxnumberxxx` adalah hasil anonimisasi dan harus dipertahankan persis. Tidak ada
# pembersihan lain, karena targetnya sendiri menentukan bentuk akhir.

# %%
PLACEHOLDER_RE = re.compile(r"xxx\w+xxx")
print("placeholder di latih:", Counter(PLACEHOLDER_RE.findall(" ".join(train["informal"]))).most_common(5))

# %% [markdown]
# ## 6. Validasi
#
# Memakai pembagian resmi. Metrik: CER dan WER (makin kecil makin baik) serta BLEU. Baseline salin teks wajib ada,
# karena pada tugas ini model sering justru memperburuk teks yang sudah benar.

# %%
def normalization_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {"cer": round(jiwer.cer(references, predictions), 4), "wer": round(jiwer.wer(references, predictions), 4),
            "bleu": round(sacrebleu.corpus_bleu(predictions, [references], force=True).score, 2)}


assert normalization_scores(["a b c d"], ["a b c d"])["cer"] == 0.0
results: list[dict] = []


def record(name: str, pred_valid: list[str], pred_test: list[str]) -> None:
    valid_scores = normalization_scores(pred_valid, valid["formal"].tolist())
    test_scores = normalization_scores(pred_test, test["formal"].tolist())
    results.append({"metode": name, "cer_valid": valid_scores["cer"], "cer_test": test_scores["cer"],
                    "wer_test": test_scores["wer"], "bleu_test": test_scores["bleu"]})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Salin teks, lalu kamus alay, lalu kamus dari data latih itu sendiri yang biasanya lebih cocok dengan domainnya.

# %%
record("salin_teks", valid["informal"].tolist(), test["informal"].tolist())
slang = load_slang_dict()
record("kamus_alay", [normalize_slang(t, slang) for t in valid["informal"]], [normalize_slang(t, slang) for t in test["informal"]])

learned: dict[str, Counter] = {}
for (word_from, word_to), count in changes.items():
    learned.setdefault(word_from, Counter())[word_to] += count
keep_counts = Counter(w for t in train["formal"] for w in t.split())
learned_map = {w: c.most_common(1)[0][0] for w, c in learned.items() if c.most_common(1)[0][1] >= 2 and c.most_common(1)[0][1] > keep_counts[w]}
combined = {**slang, **learned_map}
valid_dict = [normalize_slang(t, combined) for t in valid["informal"]]
test_dict = [normalize_slang(t, combined) for t in test["informal"]]
record("kamus_gabungan", valid_dict, test_dict)

# %% [markdown]
# ## 8. Model Utama
#
# mT5 dilatih memetakan informal ke baku. Model pretrained ini aslinya untuk meringkas, jadi fine-tune wajib.

# %%
valid_model, test_model = valid_dict, test_dict
if USE_PRETRAINED:
    tokenizer, model = load_seq2seq("mt5_sum")
    with stage("mt5_finetune"):
        model = finetune_seq2seq(model, tokenizer, train["informal"].tolist(), train["formal"].tolist(), SETTINGS["epochs"], SEED,
                                 batch_size=8, max_input=96, max_target=96, prefix=PREFIX, lr=1e-4)
    with stage("mt5_generate"):
        valid_model = generate_texts(model, tokenizer, valid["informal"].tolist(), max_input=96, max_new=96, prefix=PREFIX, beams=2)
        test_model = generate_texts(model, tokenizer, test["informal"].tolist(), max_input=96, max_new=96, prefix=PREFIX, beams=2)
    record("mt5_finetune", valid_model, test_model)
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 9. Post-processing
#
# **Pengaman.** Keluaran model yang panjangnya jauh berbeda dari masukan, atau kehilangan placeholder, dianggap
# gagal dan diganti hasil kamus. Pemilihan akhir antara kamus dan model memakai CER di valid.

# %%
def guarded(source: str, predicted: str, fallback: str) -> str:
    ratio = len(predicted.split()) / max(len(source.split()), 1)
    placeholders_ok = sorted(PLACEHOLDER_RE.findall(source)) == sorted(PLACEHOLDER_RE.findall(predicted))
    return predicted if 0.6 <= ratio <= 1.6 and placeholders_ok else fallback


valid_guard = [guarded(s, p, f) for s, p, f in zip(valid["informal"], valid_model, valid_dict)]
test_guard = [guarded(s, p, f) for s, p, f in zip(test["informal"], test_model, test_dict)]
record("model_plus_pengaman", valid_guard, test_guard)
use_model = jiwer.cer(valid["formal"].tolist(), valid_guard) < jiwer.cer(valid["formal"].tolist(), valid_dict)
final_test = test_guard if use_model else test_dict
print("dipilih dari valid:", "model dengan pengaman" if use_model else "kamus gabungan")

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Contoh hasil,** diurutkan dari kalimat dengan selisih terbesar.

# %%
examples = pd.DataFrame({"informal": test["informal"].str[:60], "baku": test["formal"].str[:60], "hasil": [t[:60] for t in final_test]})
examples["cer"] = [round(jiwer.cer(r, p), 3) if p else 1.0 for r, p in zip(test["formal"], final_test)]
examples.sort_values("cer", ascending=False).head(6)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": test.index, "baku": final_test}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("normalisasi", len(train), {r["metode"]: {"cer_test": r["cer_test"], "bleu_test": r["bleu_test"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# ISI_SETELAH_EDA
