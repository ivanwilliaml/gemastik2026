# %% [markdown]
# # 03 Clustering dengan Label Tersembunyi
#
# Teks dikelompokkan tanpa label, tapi penilaiannya memakai label tersembunyi lewat NMI atau ARI.
# Contoh: judul berita Indonesia dengan kategori disembunyikan.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Mengelompokkan teks yang label aslinya tersembunyi |
# | **Unit prediksi** | Satu judul mendapat satu cluster |
# | **Metrik** | Saat lomba tanpa label: stabilitas. Audit di sini: NMI, ARI |
# | **Masalah terbesar** | Silhouette menyesatkan: Spearman silhouette terhadap NMI -0,946. Skor silhouette tertinggi justru berasal dari cluster timpang yang hampir tidak bermakna. |
# | **Data contoh** | 5.000 judul berita, kategori asli hanya untuk audit |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta mengelompokkan teks, dan leaderboard menilai dengan NMI, ARI, V-measure,
# atau akurasi hasil pencocokan. Kalau penilaiannya coherence atau kualitas topik, pakai folder `01`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data` dan `TEXT_COL`. Di soal asli kolom `category` tidak ada, jadi hapus sel audit.
# - `N_CLUSTERS`, isi kalau soal menyebut jumlahnya. Kalau tidak, biarkan `None`.

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
TEXT_COL, HIDDEN_LABEL_COL = "title", "category"
N_CLUSTERS = None  # None: ditentukan dari stabilitas
MAX_CLUSTER_SHARE = 0.5

SETTINGS = {
    "cepat": {"k_grid": [6, 9, 12], "bag_seeds": 5},
    "penuh": {"k_grid": [5, 6, 7, 8, 9, 10, 12, 15], "bag_seeds": 20},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include stopwords

# %% include embedding

# %% [markdown]
# ## 4. EDA
#
# Muat data. Kolom kategori disisihkan dan hanya dipakai untuk audit di bagian akhir.

# %%
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("indonesian-news-title.csv")).dropna(subset=[TEXT_COL])
    return frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)


data = load_data()
hidden_labels = data.pop(HIDDEN_LABEL_COL)
print(f"baris: {len(data)}")
text_profile(data[TEXT_COL])

# %% [markdown]
# Contoh judul, kata terbanyak, dan distribusi kategori tersembunyi. Yang terakhir hanya boleh dilihat
# di latihan ini.

# %%
print(data[TEXT_COL].head(4).to_string(index=False))
words = data[TEXT_COL].str.lower().str.findall(r"[a-z]{4,}").explode()
fig, axes = plt.subplots(1, 2)
words.value_counts().head(12).sort_values().plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set(title="Kata terbanyak", xlabel="frekuensi", ylabel="kata")
hidden_labels.value_counts().plot.bar(ax=axes[1], color="#4C72B0")
axes[1].set(title="Kategori tersembunyi (audit)", xlabel="kategori", ylabel="jumlah baris")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Judul sangat pendek: median 9 kata, tanpa duplikat | Pembersihan cukup huruf kecil dan spasi, stopword tetap dipertahankan |
# | Berita corona mendominasi | Satu cluster besar wajar, tapi cluster terlalu timpang ditolak |
# | Embed + agglomerative punya silhouette 0,2272 tapi NMI 0,0057 | Pilih model dari stabilitas, bukan silhouette |

# %% [markdown]
# ## 5. Preprocessing
#
# Judul sudah bersih, jadi cukup huruf kecil dan spasi. Stopword tidak dibuang karena model embedding
# memakai kalimat utuh.

# %%
data["clean"] = data[TEXT_COL].map(clean_text)
print("duplikat setelah dibersihkan:", int(data["clean"].duplicated().sum()))

# %% [markdown]
# ## 6. Validasi
#
# Di soal asli label tidak ada, jadi pemilihan model **tidak boleh** memakai label. NMI, ARI, dan akurasi hasil
# pencocokan dihitung sebagai audit dan tidak ikut memilih. Metrik diuji dulu.

# %%
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


def matched_accuracy(truth, labels) -> float:
    """Akurasi setelah cluster dicocokkan dengan kategori memakai algoritma Hungarian."""
    truth_ids = pd.factorize(truth)[0]
    labels_ids = pd.factorize(labels)[0]
    size = max(truth_ids.max(), labels_ids.max()) + 1
    table = confusion_matrix(truth_ids, labels_ids, labels=range(size))
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(truth_ids))


assert matched_accuracy(["a", "a", "b", "b"], [1, 1, 0, 0]) == 1.0
assert matched_accuracy(["a", "a", "b", "b"], [0, 1, 0, 1]) == 0.5

results: list[dict] = []


def record(name: str, labels: np.ndarray, vectors: np.ndarray) -> None:
    results.append(
        {"model": name, "k": int(len(np.unique(labels))),
         "silhouette": round(float(silhouette_score(vectors, labels, metric="cosine",
                                                    sample_size=3000, random_state=SEED)), 4),
         "nmi_audit": round(normalized_mutual_info_score(hidden_labels, labels), 4),
         "ari_audit": round(adjusted_rand_score(hidden_labels, labels), 4),
         "v_audit": round(v_measure_score(hidden_labels, labels), 4),
         "acc_audit": round(matched_accuracy(hidden_labels, labels), 4)}
    )

# %% [markdown]
# ## 7. Baseline
#
# TF-IDF + SVD + KMeans, tanpa model dari luar.

# %%
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402

runs: dict[str, np.ndarray] = {}
with stage("baseline"):
    vectors_tfidf = tfidf_svd_embed(data["clean"].tolist())
    for k in SETTINGS["k_grid"]:
        runs[f"tfidf_kmeans_{k}"] = KMeans(k, random_state=SEED, n_init=10).fit_predict(vectors_tfidf)
        record(f"tfidf_kmeans_{k}", runs[f"tfidf_kmeans_{k}"], vectors_tfidf)
pd.DataFrame(results)

# %% [markdown]
# ## 8. Model Utama
#
# Embedding + KMeans dan agglomerative, dengan dua pilihan ruang: embedding langsung dan hasil UMAP.

# %%
import umap  # noqa: E402

with stage("embedding"):
    embeddings = embed_texts(data["clean"].tolist())
with stage("umap"):
    reduced = umap.UMAP(n_neighbors=30, n_components=10, min_dist=0.0, metric="cosine",
                        random_state=SEED).fit_transform(embeddings)

with stage("clustering"):
    for k in SETTINGS["k_grid"]:
        for space, vectors in (("embed", embeddings), ("umap", reduced)):
            name = f"{space}_kmeans_{k}"
            runs[name] = KMeans(k, random_state=SEED, n_init=10).fit_predict(vectors)
            record(name, runs[name], embeddings)
        name = f"embed_agglo_{k}"
        runs[name] = AgglomerativeClustering(k, metric="cosine", linkage="average").fit_predict(embeddings)
        record(name, runs[name], embeddings)
pd.DataFrame(results).sort_values("silhouette", ascending=False).head(10)

# %% [markdown]
# ## 9. Post-processing
#
# **Silhouette saja menyesatkan.** Agglomerative average sering membuat satu cluster raksasa dan sisanya kecil, dan
# itu justru dapat silhouette tinggi. Jadi aturan pemilihannya, tetap tanpa label:
# 1. Tolak hasil yang timpang, yaitu cluster terbesar melebihi `MAX_CLUSTER_SHARE`.
# 2. Pilih yang paling **stabil** antar seed, karena struktur asli akan muncul lagi saat seed diganti.
#
# Lalu hasil dimantapkan dengan **consensus**: label dari banyak seed dijadikan fitur, lalu di-cluster lagi.

# %%
def largest_share(labels: np.ndarray) -> float:
    return float(np.bincount(pd.factorize(labels)[0]).max() / len(labels))


def stability_score(vectors: np.ndarray, k: int, seeds=(101, 102, 103)) -> float:
    """Rata-rata ARI antar hasil KMeans dengan seed berbeda."""
    labelings = [KMeans(k, random_state=s, n_init=3).fit_predict(vectors) for s in seeds]
    pairs = [(0, 1), (0, 2), (1, 2)]
    return float(np.mean([adjusted_rand_score(labelings[a], labelings[b]) for a, b in pairs]))


selection = pd.DataFrame(results)
selection = selection[selection["model"].str.startswith(("embed_kmeans", "umap_kmeans"))].copy()
selection["terbesar"] = [largest_share(runs[m]) for m in selection["model"]]
selection = selection[selection["terbesar"] <= MAX_CLUSTER_SHARE]
with stage("stabilitas"):
    selection["stabilitas"] = [
        stability_score(reduced if m.startswith("umap") else embeddings, int(k))
        for m, k in zip(selection["model"], selection["k"])
    ]
best = selection.sort_values("stabilitas", ascending=False).iloc[0]
best_k = N_CLUSTERS or int(best["k"])
print(f"dipilih dari stabilitas: {best['model']} -> k={best_k}")
stability_table = selection[["model", "k", "silhouette", "stabilitas", "terbesar", "nmi_audit"]].round(4)


def consensus_labels(vectors: np.ndarray, k: int, n_seeds: int) -> np.ndarray:
    """KMeans pada label one-hot dari banyak seed: baris yang sering sekelompok jadi berdekatan."""
    columns = []
    for seed in range(n_seeds):
        labels = KMeans(k, random_state=seed, n_init=1).fit_predict(vectors)
        columns.append(np.eye(k, dtype=np.float32)[labels])
    return KMeans(k, random_state=SEED, n_init=10).fit_predict(np.hstack(columns))


with stage("consensus"):
    space_vectors = reduced if str(best["model"]).startswith("umap") else embeddings
    final_labels = consensus_labels(space_vectors, best_k, SETTINGS["bag_seeds"])
record("consensus_final", final_labels, embeddings)
stability_table.sort_values("stabilitas", ascending=False)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan,** diurutkan menurut NMI audit.

# %%
result_table = pd.DataFrame(results).sort_values("nmi_audit", ascending=False)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table.head(10)

# %% [markdown]
# **Apakah metrik tanpa label bisa dipercaya?** Korelasi silhouette dan stabilitas dengan NMI audit. Kalau korelasinya
# rendah atau negatif, jangan menyetel terlalu halus berdasarkan metrik itu.

# %%
correlation = result_table[["silhouette", "nmi_audit"]].corr(method="spearman").iloc[0, 1]
stability_correlation = stability_table[["stabilitas", "nmi_audit"]].corr(method="spearman").iloc[0, 1]
print(f"Spearman silhouette-NMI: {correlation:.3f} | stabilitas-NMI: {stability_correlation:.3f}")

# %% [markdown]
# **Isi tiap cluster:** kata khas dan kategori tersembunyi yang paling sering muncul.

# %%
vectorizer = TfidfVectorizer(min_df=5, max_df=0.3, token_pattern=r"\b[a-z]{3,}\b", stop_words=sorted(STOPWORDS))
tfidf_matrix = vectorizer.fit_transform(data["clean"])
vocab = np.array(vectorizer.get_feature_names_out())
rows = []
for cluster in np.unique(final_labels):
    members = final_labels == cluster
    top = np.asarray(tfidf_matrix[members].mean(0)).ravel().argsort()[::-1][:6]
    rows.append({"cluster": int(cluster), "jumlah": int(members.sum()), "kata_khas": ", ".join(vocab[top]),
                 "kategori_audit": hidden_labels[members].value_counts(normalize=True).round(2).head(2).to_dict()})
pd.DataFrame(rows).sort_values("jumlah", ascending=False)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": data.index, "cluster": final_labels}).to_csv(RESULT_DIR / "submission.csv", index=False)
final_row = result_table[result_table["model"] == "consensus_final"].iloc[0].to_dict()
run_summary("clustering", len(data),
            {"final": final_row, "spearman_silhouette_nmi": round(float(correlation), 3),
             "spearman_stabilitas_nmi": round(float(stability_correlation), 3)},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Stabilitas memilih UMAP + KMeans k=5 tanpa melihat label. Hasil konsensus: NMI audit 0,1538, jauh di atas TF-IDF + KMeans (0,0432 pada k=5).
# - Korelasi stabilitas terhadap NMI positif (0,229), silhouette negatif (-0,946).
# - Secara jujur, k=15 punya NMI lebih tinggi (0,2339). Itu hanya terlihat dari audit, jadi tidak boleh dipakai memilih.
# - NMI masih rendah. Kategori berita tidak sama dengan tema judul, jadi cluster harus diberi nama dari kata khasnya.
