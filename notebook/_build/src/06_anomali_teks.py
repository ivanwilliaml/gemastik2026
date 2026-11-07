# %% [markdown]
# # 06 Deteksi Anomali Teks
#
# Menemukan teks yang menyimpang tanpa label. Contoh: komentar MBG yang disisipi teks luar topik, supaya skornya
# bisa diukur.
#
# **Peringatan:** skor pada anomali buatan cenderung jauh lebih tinggi daripada pada anomali sungguhan. Anggap
# sebagai pembanding antar metode, bukan perkiraan kinerja asli.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Mendeteksi teks yang tidak wajar tanpa label |
# | **Unit prediksi** | Satu komentar mendapat skor anomali |
# | **Metrik** | AUC dan average precision pada injeksi sintetis |
# | **Masalah terbesar** | Tidak ada contoh anomali asli, jadi kualitas hanya bisa diukur dari teks luar topik yang disisipkan. Anomali asli bisa saja berbeda sifatnya. |
# | **Data contoh** | 4.795 komentar MBG, disisipi judul berita luar topik |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta mendeteksi teks aneh, luar topik, spam, atau hoaks tanpa contoh berlabel. Kalau
# yang dicari salinan teks yang mirip, pakai folder `07`. Kalau ada label, pakai folder `09`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`. Di soal asli, injeksi hanya untuk menilai metode, dan skor akhir dihitung pada data tanpa injeksi.
# - `CONTAMINATION`, perkiraan porsi anomali.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: embedding memakai TF-IDF + SVD, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
TEXT_COL = "Comment"
INJECT_SHARE = 0.05
CONTAMINATION = 0.05

SETTINGS = {"cepat": {"eval_seeds": [1, 2]}, "penuh": {"eval_seeds": [1, 2, 3, 4, 5]}}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include embedding

# %% [markdown]
# ## 4. EDA
#
# Muat komentar sebagai data normal dan judul berita sebagai bahan anomali buatan.

# %%
import html  # noqa: E402

HTML_RE = re.compile(r"<[^>]+>")


def tidy(text: str) -> str:
    return clean_text(HTML_RE.sub(" ", html.unescape(str(text))))


comments = pd.read_csv(find_file("comment_data.csv"))[TEXT_COL].dropna().map(tidy)
comments = comments[comments.str.split().str.len() >= 3].drop_duplicates().head(N_ROWS).reset_index(drop=True)
news = pd.read_csv(find_file("indonesian-news-title.csv"))
news = news[news["category"].isin(["oto", "travel", "food", "inet"])]["title"].dropna().map(tidy)
print(f"komentar={len(comments)} | bahan berita luar topik={len(news)}")
text_profile(comments)

# %% [markdown]
# Sebaran panjang komentar. Komentar sangat panjang sering berisi promosi atau tempelan.

# %%
lengths = comments.str.split().str.len()
fig, ax = plt.subplots()
lengths.clip(upper=150).plot.hist(bins=40, ax=ax, color="#4C72B0")
ax.set(title="Panjang komentar", xlabel="jumlah kata", ylabel="jumlah baris")
plt.tight_layout()
plt.show()
print("komentar berisi tautan:", round(100 * comments.str.contains("http|www").mean(), 1), "%")

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Komentar berisi tautan 0% | Pola tautan tidak bisa dijadikan sinyal |
# | Slang dan tanda baca bisa jadi sinyal | Teks hanya dibersihkan ringan |
# | Hasil bisa bergantung pada injeksi | Pakai beberapa seed, lalu laporkan rata-rata dan simpangan |

# %% [markdown]
# ## 5. Preprocessing
#
# Teks dibersihkan ringan dan tidak dinormalisasi. Slang, tanda seru, dan pola penulisan yang tidak biasa
# justru termasuk sinyal anomali. Embedding dihitung sekali untuk komentar dan bahan injeksi.

# %%
all_news = news.drop_duplicates().sample(n=min(1500, len(news)), random_state=SEED).reset_index(drop=True)
all_texts = comments.tolist() + all_news.tolist()
with stage("embedding"):
    all_vectors = embed_texts(all_texts)
comment_vectors = all_vectors[: len(comments)]
news_vectors = all_vectors[len(comments) :]
print("embedding:", all_vectors.shape)

# %% [markdown]
# ## 6. Validasi
#
# **Injeksi sintetis:** sebagian judul berita luar topik disisipkan, lalu dilihat apakah metode menaruhnya di
# peringkat atas. Beberapa seed injeksi dipakai, supaya hasilnya tidak bergantung pada satu undian. Metrik: AUC,
# average precision (AP), dan recall pada `CONTAMINATION` teratas.

# %%
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.metrics import roc_auc_score as auc  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402


def recall_at_share(target_flags: np.ndarray, scores: np.ndarray, share: float) -> float:
    top = np.argsort(-scores)[: int(share * len(scores))]
    return float(target_flags[top].sum() / max(target_flags.sum(), 1))


assert recall_at_share(np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0.2, 0.3]), 0.25) == 1.0


def inject(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gabungkan komentar dengan sebagian berita. Kembalikan vektor dan penanda injeksi."""
    rng = np.random.default_rng(seed)
    n_inject = int(INJECT_SHARE * len(comment_vectors))
    picks = rng.choice(len(news_vectors), n_inject, replace=False)
    vectors = np.vstack([comment_vectors, news_vectors[picks]])
    flags = np.array([0] * len(comment_vectors) + [1] * n_inject)
    order = rng.permutation(len(vectors))
    return vectors[order], flags[order]


results: list[dict] = []


def evaluate(name: str, score_fn) -> None:
    """Rata-rata skor antar seed injeksi."""
    rows = []
    for seed in SETTINGS["eval_seeds"]:
        vectors, flags = inject(seed)
        scores = score_fn(vectors, seed)
        rows.append({"auc": auc(flags, scores), "ap": average_precision_score(flags, scores),
                     "recall_atas": recall_at_share(flags, scores, CONTAMINATION)})
    frame = pd.DataFrame(rows)
    results.append({"metode": name, **frame.mean().round(4).to_dict(), "auc_sd": round(frame["auc"].std(), 4)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Jarak ke rata-rata seluruh data: makin jauh, makin aneh. Sederhana dan tanpa parameter.

# %%
def centroid_distance(vectors: np.ndarray, _seed: int) -> np.ndarray:
    return 1 - vectors @ (vectors.mean(0) / np.linalg.norm(vectors.mean(0)))


evaluate("jarak_ke_pusat", centroid_distance)

# %% [markdown]
# ## 8. Model Utama
#
# **IsolationForest** mengisolasi titik yang mudah dipisahkan. **LOF** melihat kerapatan lokal, jadi lebih cocok
# kalau data normal punya beberapa kelompok. **kNN** memakai jarak ke tetangga terdekat.

# %%
def isolation_scores(vectors: np.ndarray, seed: int) -> np.ndarray:
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(vectors)
    return -model.score_samples(vectors)


def lof_scores(vectors: np.ndarray, _seed: int) -> np.ndarray:
    model = LocalOutlierFactor(n_neighbors=30, metric="cosine", n_jobs=-1).fit(vectors)
    return -model.negative_outlier_factor_


def knn_scores(vectors: np.ndarray, _seed: int) -> np.ndarray:
    model = LocalOutlierFactor(n_neighbors=10, metric="cosine", n_jobs=-1).fit(vectors)
    distance, _ = model.kneighbors(vectors)
    return distance.mean(1)


with stage("model"):
    evaluate("isolation_forest", isolation_scores)
    evaluate("lof", lof_scores)
    evaluate("knn", knn_scores)

# %% [markdown]
# ## 9. Post-processing
#
# Gabungkan metode dengan rata-rata peringkat, karena skala skornya berbeda. Rata-rata tanpa bobot dipakai
# karena tidak ada label untuk memfit bobot.

# %%
METHODS = {"isolation_forest": isolation_scores, "lof": lof_scores, "knn": knn_scores}


def rank_average(vectors: np.ndarray, seed: int) -> np.ndarray:
    ranks = [pd.Series(fn(vectors, seed)).rank(pct=True).to_numpy() for fn in METHODS.values()]
    return np.mean(ranks, axis=0)


evaluate("gabungan_peringkat", rank_average)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.** Selisih di dalam `auc_sd` jangan dianggap perbaikan nyata.

# %%
result_table = pd.DataFrame(results).sort_values("auc", ascending=False)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Anomali sungguhan.** Metode dijalankan pada komentar asli **tanpa injeksi**. Baca hasil teratas secara manual.
# Ini satu-satunya gambaran kualitas yang sebenarnya.

# %%
real_scores = rank_average(comment_vectors, SEED)
top_real = pd.DataFrame({"komentar": comments.str[:110], "skor": real_scores.round(3)}).sort_values("skor", ascending=False)
top_real.head(10)

# %% [markdown]
# **Sebaran skor** untuk memilih batas.

# %%
threshold = np.quantile(real_scores, 1 - CONTAMINATION)
fig, ax = plt.subplots()
ax.hist(real_scores, bins=50, color="#4C72B0")
ax.axvline(threshold, color="#C44E52", ls="--", label=f"batas {threshold:.2f}")
ax.set(title="Sebaran skor anomali", xlabel="skor", ylabel="jumlah komentar")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": comments.index, "skor": real_scores.round(4), "anomali": (real_scores >= threshold).astype(int)}).to_csv(
    RESULT_DIR / "submission.csv", index=False
)
run_summary("anomali_teks", len(comments), {r["metode"]: {"auc": r["auc"], "ap": r["ap"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - kNN pada embedding terbaik: AUC 0,946 dan average precision 0,5257.
# - Gabungan peringkat sedikit lebih rendah (AUC 0,9414), tapi simpangannya lebih kecil (0,0047 dibanding 0,0099). Kalau metriknya AP, kNN jelas lebih baik.
# - Isolation Forest paling lemah di sini (AUC 0,8729), walaupun paling populer.
# - Komentar peringkat atas banyak berbahasa Inggris atau tidak nyambung. Ini cocok dipakai untuk membersihkan data sebelum modelling.
