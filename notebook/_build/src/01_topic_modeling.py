# %% [markdown]
# # 01 Topic Modeling
#
# Data tanpa label, tujuannya menemukan tema. Contoh: komentar YouTube tentang program MBG.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Menemukan topik tanpa label |
# | **Unit prediksi** | Satu komentar mendapat satu topik |
# | **Metrik** | NPMI, diversity, dan stabilitas antar seed (ARI) |
# | **Masalah terbesar** | Komentar pendek dan kotor: median 17 kata, maksimum 1.446 kata, 17,8% berisi tag HTML. Tidak ada label untuk menilai hasil. |
# | **Data contoh** | 5.000 komentar YouTube tentang MBG |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta menemukan topik atau tema dari teks tanpa label. Metriknya biasanya
# coherence (NPMI), diversity, silhouette, atau penilaian kualitas topik saat presentasi.
#
# **Kalau ada label tersembunyi** dan penilaiannya NMI atau ARI, pakai folder `03`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data` dan `TEXT_COL` di bagian 3 dan 4.
# - `DOMAIN_STOPWORDS`, isi dengan kata yang muncul di hampir semua dokumen.
# - `K_GRID`, sesuaikan dengan jumlah topik yang masuk akal atau yang diminta soal.

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
DOMAIN_STOPWORDS = {"mbg", "makan", "bergizi", "gratis", "program", "prof", "gema"}
TOP_WORDS = 10

SETTINGS = {
    "cepat": {"k_grid": [6, 10, 15], "stability_seeds": [1, 2]},
    "penuh": {"k_grid": [5, 6, 8, 10, 12, 15, 20, 25], "stability_seeds": [1, 2, 3, 4, 5]},
}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include stopwords

# %% include embedding

# %% [markdown]
# ## 4. EDA
#
# Muat data dan lihat kualitas teksnya.

# %%
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("comment_data.csv"))
    return frame.dropna(subset=[TEXT_COL]).head(N_ROWS).reset_index(drop=True)


data = load_data()
print(f"baris: {len(data)}")
text_profile(data[TEXT_COL])

# %% [markdown]
# Kata terbanyak sebelum dibersihkan, dan apakah ada komentar yang sama persis berulang.

# %%
raw_words = data[TEXT_COL].astype(str).str.lower().str.findall(r"[a-z]{3,}").explode()
top_raw = raw_words[~raw_words.isin(STOPWORDS)].value_counts().head(15)
top_repeated = data[TEXT_COL].astype(str).str[:80].value_counts().head(5)
print("Komentar paling sering berulang:")
print(top_repeated.to_string())
print("\nAda tag HTML:", round(100 * data[TEXT_COL].astype(str).str.contains("<br>|<a ").mean(), 1), "%")

fig, axes = plt.subplots(1, 2)
top_raw.sort_values().plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set(title="Kata terbanyak", xlabel="frekuensi", ylabel="kata")
data[TEXT_COL].astype(str).str.split().str.len().clip(upper=120).plot.hist(bins=40, ax=axes[1], color="#4C72B0")
axes[1].set(title="Panjang komentar", xlabel="jumlah kata", ylabel="jumlah baris")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | 17,8% komentar berisi tag HTML | HTML dibuang, supaya tidak jadi kata kunci topik |
# | Ada komentar yang sama persis dan sangat pendek, misalnya `Mantap` | Duplikat disatukan dan komentar pendek dibuang, sisa 4.456 baris |
# | Median 17 kata, jadi jarang muncul kata yang sama | Pakai dua versi teks: `clean` untuk embedding, `words` untuk kata kunci |
# | HDBSCAN meninggalkan 35,2% outlier | Outlier ditempelkan ke cluster terdekat, dan NMF tetap dibandingkan |

# %% [markdown]
# ## 5. Preprocessing
#
# Dua versi teks: `clean` untuk embedding, dan `words` tanpa stopword untuk kata kunci topik. HTML dan
# komentar yang terlalu pendek dibuang, sedangkan duplikat dijadikan satu supaya tidak membentuk
# topik semu.

# %%
import html  # noqa: E402

HTML_RE = re.compile(r"<[^>]+>")
slang = load_slang_dict()
all_stopwords = STOPWORDS | DOMAIN_STOPWORDS

data["clean"] = data[TEXT_COL].astype(str).map(lambda t: clean_text(HTML_RE.sub(" ", html.unescape(t))))
data["formal"] = data["clean"].map(lambda t: normalize_slang(t, slang))
data["words"] = data["formal"].map(
    lambda t: " ".join(w for w in re.findall(r"[a-z]{3,}", t) if w not in all_stopwords)
)
before = len(data)
data = data[data["words"].str.split().str.len() >= 3].drop_duplicates("clean").reset_index(drop=True)
print(f"baris sebelum={before}, sesudah={len(data)}")
data[["clean", "words"]].head(3)

# %% [markdown]
# ## 6. Validasi
#
# Tidak ada label, jadi yang dinilai adalah kualitas topik. Metrik ditulis sendiri dan diuji dulu:
# - **NPMI**: kata dalam satu topik sering muncul bersama atau tidak. Rentang -1 sampai 1.
# - **Diversity**: bagian kata kunci yang unik antar topik.
# - **Silhouette**: pemisahan cluster, selalu dihitung di ruang embedding yang sama supaya adil.
#
# Satu angka saja tidak cukup, karena coherence bisa bagus padahal silhouette mendekati nol.

# %%
from scipy.sparse import csr_matrix  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

vectorizer = CountVectorizer(min_df=5, max_df=0.2, token_pattern=r"\b[a-z]{3,}\b")
counts = vectorizer.fit_transform(data["words"])
vocab = np.array(vectorizer.get_feature_names_out())
presence = (counts > 0).astype(np.int32)
print(f"kosakata: {len(vocab)} kata")


def npmi_coherence(topics: list[list[int]], doc_word) -> float:
    """Rata-rata NPMI pasangan kata kunci per topik, dihitung dari kemunculan dalam dokumen."""
    n_docs = doc_word.shape[0]
    together = (doc_word.T @ doc_word).tocsr()
    doc_freq = np.asarray(doc_word.sum(0)).ravel()
    scores = []
    for words in topics:
        for a, word_a in enumerate(words):
            for word_b in words[a + 1 :]:
                joint = together[word_a, word_b] / n_docs
                if joint == 0:
                    scores.append(-1.0)
                    continue
                pmi = np.log(joint / (doc_freq[word_a] / n_docs * doc_freq[word_b] / n_docs))
                scores.append(pmi / -np.log(joint))
    return float(np.mean(scores))


def topic_diversity(topics: list[list[int]]) -> float:
    flat = [w for words in topics for w in words]
    return len(set(flat)) / len(flat)


_toy = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int32)
assert abs(npmi_coherence([[0, 1]], csr_matrix(_toy)) - 1.0) < 1e-9
assert topic_diversity([[0, 1], [1, 2]]) == 0.75

# %% [markdown]
# **Embedding** dihitung sekali di sini, karena dipakai oleh semua model untuk menghitung silhouette.

# %%
with stage("embedding"):
    embeddings = embed_texts(data["clean"].tolist())
print("embedding:", embeddings.shape)

# %% [markdown]
# **Kata kunci c-TF-IDF** dipakai untuk semua model supaya perbandingannya adil. Kata yang sering di
# satu topik tapi jarang di topik lain mendapat skor tinggi.

# %%
def ctfidf_topics(labels: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """Kembalikan id topik dan kata kuncinya. Label -1 (outlier) dilewati."""
    topic_ids = [t for t in np.unique(labels) if t != -1]
    per_topic = np.vstack([np.asarray(counts[labels == t].sum(0)).ravel() for t in topic_ids])
    term_freq = per_topic / np.maximum(per_topic.sum(1, keepdims=True), 1)
    avg_words = per_topic.sum() / len(topic_ids)
    idf = np.log(1 + avg_words / np.maximum(per_topic.sum(0), 1))
    scores = term_freq * idf
    return topic_ids, [list(np.argsort(-row)[:TOP_WORDS]) for row in scores]


def evaluate(name: str, k: int, labels: np.ndarray) -> dict:
    """Hitung semua metrik untuk satu hasil topik."""
    _, topics = ctfidf_topics(labels)
    silhouette = silhouette_score(embeddings, labels, metric="cosine",
                                  sample_size=min(3000, len(labels)), random_state=SEED)
    return {"model": name, "k": k, "npmi": round(npmi_coherence(topics, presence), 4),
            "diversity": round(topic_diversity(topics), 3),
            "silhouette": round(float(silhouette), 4)}

# %% [markdown]
# ## 7. Baseline
#
# Dua pembanding klasik: NMF pada TF-IDF, dan LDA pada hitungan kata. Keduanya tidak butuh model dari
# luar, jadi tetap bisa dipakai kalau pretrained dilarang.

# %%
from sklearn.decomposition import NMF, LatentDirichletAllocation  # noqa: E402
from sklearn.feature_extraction.text import TfidfTransformer  # noqa: E402

tfidf_counts = TfidfTransformer(sublinear_tf=True).fit_transform(counts)


def nmf_labels(k: int, seed: int) -> np.ndarray:
    model = NMF(k, random_state=seed, init="nndsvda", max_iter=400)
    return model.fit_transform(tfidf_counts).argmax(1)


def lda_labels(k: int, seed: int) -> np.ndarray:
    model = LatentDirichletAllocation(k, random_state=seed, learning_method="online")
    return model.fit_transform(counts).argmax(1)


results: list[dict] = []
runs: dict[tuple[str, int], np.ndarray] = {}
with stage("baseline"):
    for k in SETTINGS["k_grid"]:
        for name, fit_fn in (("nmf", nmf_labels), ("lda", lda_labels)):
            runs[(name, k)] = fit_fn(k, SEED)
            results.append(evaluate(name, k, runs[(name, k)]))
pd.DataFrame(results).sort_values("npmi", ascending=False).head(6)

# %% [markdown]
# ## 8. Model Utama
#
# **Embedding + UMAP + clustering + c-TF-IDF.** Embedding menangkap makna walau katanya berbeda,
# UMAP mengecilkan dimensi, lalu clustering membentuk topik. KMeans memaksa semua dokumen masuk topik,
# sedangkan HDBSCAN menentukan jumlah topik sendiri dan menandai sisanya sebagai outlier.

# %%
import umap  # noqa: E402
from sklearn.cluster import HDBSCAN, KMeans  # noqa: E402


def reduce_embeddings(seed: int) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=seed)
    return reducer.fit_transform(embeddings)


with stage("umap"):
    reduced = reduce_embeddings(SEED)
print("umap:", reduced.shape)

# %% [markdown]
# Coba KMeans di semua nilai k, dan HDBSCAN yang outlier-nya ditempelkan ke topik terdekat.

# %%
def assign_outliers(labels: np.ndarray) -> np.ndarray:
    """Tempelkan dokumen outlier ke centroid topik terdekat."""
    labels = labels.copy()
    topic_ids = [t for t in np.unique(labels) if t != -1]
    centroids = np.vstack([embeddings[labels == t].mean(0) for t in topic_ids])
    outliers = np.where(labels == -1)[0]
    if len(outliers):
        nearest = (embeddings[outliers] @ centroids.T).argmax(1)
        labels[outliers] = np.array(topic_ids)[nearest]
    return labels


with stage("clustering"):
    for k in SETTINGS["k_grid"]:
        runs[("embed_kmeans", k)] = KMeans(k, random_state=SEED, n_init=10).fit_predict(reduced)
        results.append(evaluate("embed_kmeans", k, runs[("embed_kmeans", k)]))
    min_cluster = max(15, len(data) // 150)
    raw_hdbscan = HDBSCAN(min_cluster_size=min_cluster, min_samples=5).fit_predict(reduced)
    outlier_share = float((raw_hdbscan == -1).mean())
    hdbscan_labels = assign_outliers(raw_hdbscan)
    hdbscan_k = len(np.unique(hdbscan_labels))
    runs[("embed_hdbscan", hdbscan_k)] = hdbscan_labels
    results.append(evaluate("embed_hdbscan", hdbscan_k, hdbscan_labels))
print(f"outlier HDBSCAN sebelum ditempelkan: {outlier_share:.1%}")
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table.sort_values("npmi", ascending=False).head(10)

# %% [markdown]
# ## 9. Post-processing
#
# Pilih model dari skor gabungan peringkat NPMI, diversity, dan silhouette. Peringkat dipakai karena
# skala ketiganya berbeda. Topik yang terlalu kecil lalu digabung ke topik terdekat.

# %%
result_table["skor"] = result_table[["npmi", "diversity", "silhouette"]].rank(pct=True).mean(axis=1)
best = result_table.sort_values("skor", ascending=False).iloc[0]
best_key = (best["model"], int(best["k"]))
print("dipilih:", best_key)


def merge_small_topics(labels: np.ndarray, min_share: float = 0.01) -> np.ndarray:
    """Gabungkan topik di bawah min_share ke centroid topik besar terdekat."""
    labels = labels.copy()
    sizes = pd.Series(labels).value_counts(normalize=True)
    big = sizes[sizes >= min_share].index.to_numpy()
    centroids = np.vstack([embeddings[labels == t].mean(0) for t in big])
    for topic in sizes[sizes < min_share].index:
        members = labels == topic
        labels[members] = big[(embeddings[members].mean(0) @ centroids.T).argmax()]
    return labels


final_labels = merge_small_topics(runs[best_key])
topic_ids, final_topics = ctfidf_topics(final_labels)
final_scores = evaluate("final", len(topic_ids), final_labels)
final_scores

# %% [markdown]
# ## 10. Post-analysis
#
# **Isi tiap topik:** kata kunci, ukuran, dan komentar yang paling mewakili.

# %%
rows = []
for topic, words in zip(topic_ids, final_topics):
    members = np.where(final_labels == topic)[0]
    centroid = embeddings[members].mean(0)
    closest = members[np.argmax(embeddings[members] @ centroid)]
    rows.append({"topik": int(topic), "jumlah": len(members),
                 "kata_kunci": ", ".join(vocab[words][:7]), "contoh": data["clean"].iloc[closest][:90]})
topic_table = pd.DataFrame(rows).sort_values("jumlah", ascending=False)
topic_table.to_csv(RESULT_DIR / "topik.csv", index=False)
topic_table

# %% [markdown]
# **Sebaran ukuran topik** untuk melihat apakah ada satu topik yang mendominasi.

# %%
fig, ax = plt.subplots()
ax.bar(topic_table["topik"].astype(str), topic_table["jumlah"], color="#4C72B0")
ax.set(title="Ukuran tiap topik", xlabel="id topik", ylabel="jumlah komentar")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Stabilitas.** Metode terpilih dijalankan ulang dengan seed lain, termasuk seed UMAP. ARI mendekati
# 1 berarti stabil, mendekati 0 berarti hasilnya banyak dipengaruhi kebetulan.

# %%
def refit_labels(seed: int) -> np.ndarray:
    name, k = best_key
    if name == "nmf":
        return nmf_labels(k, seed)
    if name == "lda":
        return lda_labels(k, seed)
    new_reduced = reduce_embeddings(seed)
    if name == "embed_kmeans":
        return KMeans(k, random_state=seed, n_init=10).fit_predict(new_reduced)
    return assign_outliers(HDBSCAN(min_cluster_size=min_cluster, min_samples=5).fit_predict(new_reduced))


with stage("stabilitas"):
    stability = [adjusted_rand_score(runs[best_key], refit_labels(s)) for s in SETTINGS["stability_seeds"]]
stability_mean = float(np.mean(stability))
print(f"ARI antar seed: {np.round(stability, 3)} | rata-rata {stability_mean:.3f}")

# %% [markdown]
# ## 11. Keluaran
#
# Label topik per dokumen dan ringkasan run.

# %%
submission = pd.DataFrame({"id": data.index, "topic": final_labels})
submission.to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary(
    "topic_modeling",
    len(data),
    {"final": final_scores, "dipilih": f"{best_key[0]}_{best_key[1]}",
     "stabilitas_ari": round(stability_mean, 3), "outlier_hdbscan": round(outlier_share, 3)},
    needs_gpu=USE_PRETRAINED,
)

# %% [markdown]
# **Poin untuk presentasi**
#
# - NMF dengan 5 topik terpilih: NPMI 0,1727 dan diversity 0,86, jauh di atas embedding + HDBSCAN (NPMI 0,0814).
# - Stabil: ARI antar seed rata-rata 0,984.
# - Lima topik yang muncul: dapur dan harga, pemerintah, korupsi, anak sekolah dan gizi, serta apresiasi.
# - Model paling modern belum tentu terbaik untuk komentar pendek. Keputusan diambil dari metrik, bukan dari popularitas metode.
