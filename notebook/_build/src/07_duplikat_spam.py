# %% [markdown]
# # 07 Near-Duplicate, Spam, dan Bot
#
# Menemukan teks salinan dan akun yang mencurigakan. Contoh: komentar TikTok tentang MBG. Notebook ini jalan di CPU.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Menemukan komentar kembar, spam, dan akun mencurigakan |
# | **Unit prediksi** | Pasangan komentar, lalu grup, lalu akun |
# | **Metrik** | Recall dan presisi pada duplikat sintetis |
# | **Masalah terbesar** | Kembar sering sudah diubah sedikit: pencocokan persis hanya menangkap 48,85% salinan. Banyak juga komentar pendek yang sama secara wajar, seperti emoji. |
# | **Data contoh** | 5.000 komentar TikTok MBG, nama akun disamarkan |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta mendeteksi buzzer, salinan komentar, atau spam. Juga dipakai sebagai langkah awal
# di task lain, karena duplikat bisa membocorkan data antar fold.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `TEXT_COL`, `USER_COL`.
# - `SPAM_PATTERNS`.
# - Tanpa kolom pengguna, hapus bagian skor akun di bagian 10.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = False  # notebook ini tidak butuh model dari luar
SEED = 42
N_ROWS = 5000
TEXT_COL, USER_COL = "comment", "username"
SPAM_PATTERNS = r"http|www\.|wa\.me|whatsapp|promo|diskon|slot|gacor|cek bio|klik link"
INJECT_SHARE = 0.1

SETTINGS = {
    "cepat": {"thresholds": [0.7, 0.8, 0.9], "eval_seeds": [1]},
    "penuh": {"thresholds": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "eval_seeds": [1, 2, 3]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% [markdown]
# ## 4. EDA
#
# Muat data, lalu lihat komentar kembar persis dan akun yang paling aktif.

# %%
def load_data() -> pd.DataFrame:
    frame = pd.read_excel(find_file("mbg_tiktokcomment.xlsx"))
    frame = frame.dropna(subset=[TEXT_COL])
    frame[TEXT_COL] = frame[TEXT_COL].astype(str)
    frame = frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)
    frame[USER_COL] = "akun_" + pd.Series(pd.factorize(frame[USER_COL])[0]).astype(str)  # samarkan nama akun
    return frame


data = load_data()
data["clean"] = data[TEXT_COL].map(clean_text)
print(f"baris={len(data)} | akun unik={data[USER_COL].nunique()}")
print("akun teraktif:", data[USER_COL].value_counts().head(5).to_dict())
print("komentar berulang:", data["clean"].value_counts().head(5).to_dict())
print("cocok pola spam:", round(100 * data["clean"].str.contains(SPAM_PATTERNS).mean(), 2), "%")
text_profile(data[TEXT_COL])

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | 7,8% duplikat persis, 36,9% berisi emoji | Komentar yang terlalu pendek tidak dicek, sisa 4.349 baris |
# | Pola spam hanya 0,18% | Spam kata kunci bukan masalah utama, fokus ke duplikat |
# | Akun unik 4.727 dari 5.000 baris | Skor akun lemah di data ini, karena hampir semua akun hanya berkomentar sekali |

# %% [markdown]
# ## 5. Preprocessing
#
# Dua kunci pencocokan: `kunci` hanya huruf dan angka untuk kembar persis, dan char n-gram untuk kembar yang
# sedikit diubah. Karakter berulang dirapikan, jadi `bagusss` sama dengan `bagus`.

# %%
def match_key(text: str) -> str:
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return re.sub(r"[^a-z0-9]", "", text)


data["kunci"] = data["clean"].map(match_key)
usable = data[data["kunci"].str.len() >= 8].reset_index(drop=True)
print(f"baris cukup panjang untuk dicek: {len(usable)} dari {len(data)}")

# %% [markdown]
# ## 6. Validasi
#
# Tidak ada label duplikat, jadi dibuat **injeksi**: sebagian komentar disalin lalu diubah sedikit. Metriknya
# recall (salinan yang ketemu) dan presisi pasangan (pasangan yang dilaporkan memang salinan).

# %%
def mutate(text: str, rng: np.random.Generator) -> str:
    """Salin dengan perubahan kecil: hapus huruf, tukar kata, atau tambah emoji dan tanda."""
    words = text.split()
    choice = rng.integers(0, 3)
    if choice == 0 and len(words) > 2:
        i = rng.integers(0, len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    elif choice == 1:
        i = rng.integers(0, len(words))
        if len(words[i]) > 3:
            words[i] = words[i][:-1]
    else:
        words.append(str(rng.choice(["!!", "😂", "..", "🙏"])))
    return " ".join(words)


def build_eval(seed: int) -> tuple[list[str], set[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    texts = usable["clean"].tolist()
    sources = rng.choice(len(texts), int(INJECT_SHARE * len(texts)), replace=False)
    truth = set()
    for source in sources:
        truth.add((int(source), len(texts)))
        texts.append(mutate(texts[source], rng))
    return texts, truth


def pair_scores(found: set[tuple[int, int]], truth: set[tuple[int, int]], n_original: int) -> dict[str, float]:
    """Skor hanya memperhatikan pasangan yang melibatkan baris injeksi, karena hanya itu yang jawabannya diketahui."""
    relevant = {p for p in found if p[1] >= n_original}
    hits = relevant & truth
    return {"recall": round(len(hits) / max(len(truth), 1), 4), "presisi": round(len(hits) / max(len(relevant), 1), 4)}


assert pair_scores({(0, 5), (1, 2)}, {(0, 5), (3, 6)}, 5) == {"recall": 0.5, "presisi": 1.0}
results: list[dict] = []

# %% [markdown]
# ## 7. Baseline
#
# Kembar persis setelah dibersihkan. Tanpa kesalahan, tapi tidak menangkap salinan yang diubah.

# %%
def exact_pairs(texts: list[str]) -> set[tuple[int, int]]:
    seen: dict[str, int] = {}
    pairs = set()
    for i, text in enumerate(texts):
        key = match_key(text)
        if key in seen:
            pairs.add((seen[key], i))
        else:
            seen[key] = i
    return pairs


for seed in SETTINGS["eval_seeds"]:
    texts, truth = build_eval(seed)
    results.append({"metode": "kembar_persis", "seed": seed, **pair_scores(exact_pairs(texts), truth, len(usable))})
pd.DataFrame(results)

# %% [markdown]
# ## 8. Model Utama
#
# Char TF-IDF + tetangga terdekat. Char n-gram tahan terhadap salah ketik dan perubahan kecil.

# %%
def neighbor_table(texts: list[str], n_neighbors: int = 6):
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit_transform(texts)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1).fit(vectors)
    return model.kneighbors(vectors)


def char_pairs(distance, index, threshold: float) -> set[tuple[int, int]]:
    pairs = set()
    for i, (row_distance, row_index) in enumerate(zip(distance, index)):
        for dist, j in zip(row_distance, row_index):
            if int(j) != i and dist <= 1 - threshold:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


with stage("tetangga"):
    for seed in SETTINGS["eval_seeds"]:
        texts, truth = build_eval(seed)
        distance, index = neighbor_table(texts)
        for threshold in SETTINGS["thresholds"]:
            found = char_pairs(distance, index, threshold)
            results.append({"metode": f"char_tfidf_{threshold}", "seed": seed, **pair_scores(found, truth, len(usable))})
result_table = pd.DataFrame(results).groupby("metode")[["recall", "presisi"]].mean().round(4)
result_table["f1"] = (2 * result_table["recall"] * result_table["presisi"] /
                      (result_table["recall"] + result_table["presisi"]).clip(lower=1e-9)).round(4)
result_table.sort_values("f1", ascending=False)

# %% [markdown]
# ## 9. Post-processing
#
# Pilih threshold dengan F1 tertinggi dari injeksi, terapkan ke data asli, lalu satukan pasangan menjadi grup
# dengan union-find. Satu grup berarti satu kampanye teks yang mungkin sama.

# %%
result_table.to_csv(RESULT_DIR / "perbandingan.csv")
candidates = result_table[result_table.index.str.startswith("char")]
best_threshold = float(candidates["f1"].idxmax().rsplit("_", 1)[1])
print("threshold terpilih:", best_threshold)
usable["grup"] = near_duplicate_groups(usable["clean"].tolist(), threshold=best_threshold)
group_sizes = usable["grup"].value_counts()
print(f"grup berisi lebih dari satu: {(group_sizes > 1).sum()} | baris di dalamnya: {group_sizes[group_sizes > 1].sum()}")

# %% [markdown]
# ## 10. Post-analysis
#
# **Grup terbesar.** Baca isinya: komentar umum seperti `mantap` bukan bukti kampanye.

# %%
rows = []
for group, size in group_sizes.head(10).items():
    members = usable[usable["grup"] == group]
    rows.append({"grup": int(group), "jumlah": int(size), "akun_unik": members[USER_COL].nunique(),
                 "contoh": members["clean"].iloc[0][:80]})
pd.DataFrame(rows)

# %% [markdown]
# **Skor kecurigaan akun.** Gabungan aktivitas, porsi komentar duplikat, dan porsi pola spam. Nama akun sudah
# disamarkan. Ini daftar untuk diperiksa, bukan vonis akun tersebut bot.

# %%
usable["duplikat"] = usable["grup"].map(group_sizes) > 1
usable["spam"] = usable["clean"].str.contains(SPAM_PATTERNS)
account = usable.groupby(USER_COL).agg(
    komentar=("clean", "size"), porsi_duplikat=("duplikat", "mean"), porsi_spam=("spam", "mean"),
    video_unik=("video_id", "nunique"),
)
account = account[account["komentar"] >= 2]
account["skor"] = account[["komentar", "porsi_duplikat", "porsi_spam"]].rank(pct=True).mean(axis=1).round(3)
account.sort_values("skor", ascending=False).head(8).round(3)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
usable[["grup", "duplikat", "spam"]].rename_axis("id").to_csv(RESULT_DIR / "submission.csv")
best_row = result_table.loc[candidates["f1"].idxmax()].to_dict()
run_summary("duplikat_spam", len(data), {"threshold": best_threshold, "injeksi": best_row}, needs_gpu=False)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Char TF-IDF dengan threshold 0,8: F1 0,9439, sedangkan kembar persis 0,6454. Berjalan 17 detik di CPU.
# - Ditemukan 185 grup berisi 397 baris.
# - Grup terbesar seperti `alhamdulillah mbg di smp ku` muncul dari 10 akun berbeda. Ini indikasi, bukan bukti bot.
# - Skor sintetis hanya berlaku untuk perubahan yang kita simulasikan, jadi periksa contoh grup secara manual.
