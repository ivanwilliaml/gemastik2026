# %% [markdown]
# # 22 Retrieval dan Ranking
#
# Mencari dokumen yang paling relevan untuk sebuah kueri. Contoh: ringkasan IndoSum sebagai kueri, dan artikel
# aslinya sebagai jawaban.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Mencari dokumen paling relevan |
# | **Unit prediksi** | Satu kueri mendapat daftar dokumen terurut |
# | **Metrik** | MRR@10, Recall@k, dan NDCG@10 |
# | **Masalah terbesar** | 98,6% artikel lebih panjang dari 128 kata sehingga terpotong untuk embedding, dan kueri di sini sintetis (5 kata acak dari ringkasan), jadi masih memihak pencocokan kata persis. |
# | **Data contoh** | IndoSum: 4.449 artikel dan 1.000 kueri buatan |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta mencari dokumen, mencocokkan keluhan dengan kebijakan, atau meranking hasil.
# Metriknya biasanya MRR, NDCG, MAP@k, atau Recall@k.
#
# **Yang perlu diganti kalau datanya beda:**
# - `corpus`, `queries`, dan `gold`. Kalau jawaban lebih dari satu, sesuaikan fungsi metrik.
# - `RERANK_TOP`, makin besar makin akurat tapi makin lambat.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya BM25, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
KS = [1, 5, 10]
RRF_K = 60

SETTINGS = {
    "cepat": {"n_queries": 300, "rerank_top": 20, "query_word_limit": 5},
    "penuh": {"n_queries": 1000, "rerank_top": 50, "query_word_limit": 5},
}[MODE]

seed_everything(SEED)
install_offline("rank-bm25")
print(SETTINGS)

# %% include text_utils

# %% include embedding

# %% include indosum

# %% [markdown]
# ## 4. EDA
#
# Muat korpus. Ringkasan IndoSum banyak yang menyalin artikel, jadi kueri dibuat dari **lima kata acak** ringkasan,
# supaya lebih mirip kueri pencarian. Dengan 12 kata pertama, BM25 saja sudah mencapai MRR 0,98.

# %%
data = load_indosum("test.01.jsonl", N_ROWS)
data = pd.concat([data, load_indosum("dev.01.jsonl", N_ROWS)], ignore_index=True).head(N_ROWS)
data = data[data["first_summary"].str.split().str.len() >= 8].reset_index(drop=True)
print(f"artikel: {len(data)}")
print(data["category"].value_counts().to_dict())
text_profile(data["article"])

# %% [markdown]
# Panjang artikel. Model hanya membaca bagian awal, jadi perlu dilihat seberapa banyak yang terpotong.

# %%
lengths = data["article"].str.split().str.len()
fig, ax = plt.subplots()
lengths.clip(upper=800).plot.hist(bins=40, ax=ax, color="#4C72B0")
ax.set(title="Panjang artikel", xlabel="jumlah kata", ylabel="jumlah artikel")
plt.tight_layout()
plt.show()
print(f"artikel lebih dari 128 kata: {(lengths > 128).mean():.1%}")

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Dengan 12 kata pertama, BM25 sudah terlalu mudah (MRR 0,98) | Kueri dibuat lebih sulit dari 5 kata acak |
# | 98,6% artikel lebih dari 128 kata | Embedding hanya membaca awal artikel, BM25 tetap membaca seluruh artikel |
# | Kategori `inspirasi` kecil | Laporkan skor per kategori dengan jumlahnya |

# %% [markdown]
# ## 5. Preprocessing
#
# Untuk BM25: huruf kecil dan kata saja. Untuk embedding: awal artikel, karena model membaca maksimal 128 token.
# Kueri berasal dari kata acak di kalimat pertama ringkasan.

# %%
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


rng = np.random.default_rng(SEED)
query_index = rng.choice(len(data), SETTINGS["n_queries"], replace=False)


def make_query(summary: str) -> str:
    """Ambil beberapa kata isi secara acak, dengan urutan tetap."""
    words = [w for w in summary.lower().split() if len(w) > 2]
    picks = sorted(rng.choice(len(words), min(SETTINGS["query_word_limit"], len(words)), replace=False))
    return " ".join(words[i] for i in picks)


queries = [make_query(data["first_summary"].iloc[i]) for i in query_index]
gold = query_index.copy()
corpus_tokens = [tokenize(t) for t in data["article"]]
corpus_text = data["article"].map(lambda t: " ".join(t.split()[:150]).lower()).tolist()
print("contoh kueri:", queries[0])

# %% [markdown]
# ## 6. Validasi
#
# Tiap kueri punya satu dokumen benar. Metriknya MRR@10, Recall@k, dan NDCG@10. Metrik diuji dulu.

# %%
def retrieval_scores(rankings: np.ndarray, answers: np.ndarray) -> dict[str, float]:
    """rankings: indeks dokumen terurut per kueri."""
    position = np.array([np.where(row == a)[0][0] if a in row else 10**6 for row, a in zip(rankings, answers)])
    output = {"mrr@10": float(np.mean(np.where(position < 10, 1 / (position + 1), 0)))}
    for k in KS:
        output[f"recall@{k}"] = float(np.mean(position < k))
    output["ndcg@10"] = float(np.mean(np.where(position < 10, 1 / np.log2(position + 2), 0)))
    return {k: round(v, 4) for k, v in output.items()}


_check = retrieval_scores(np.array([[1, 0], [0, 1]]), np.array([1, 1]))
assert _check["mrr@10"] == 0.75 and _check["recall@1"] == 0.5
results: list[dict] = []


def record(name: str, rankings: np.ndarray) -> None:
    results.append({"metode": name, **retrieval_scores(rankings, gold)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# BM25, metode klasik yang sering sulit dikalahkan untuk kueri dengan kata yang persis sama.

# %%
from rank_bm25 import BM25Okapi  # noqa: E402

TOP_N = 100
with stage("bm25"):
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = np.vstack([bm25.get_scores(tokenize(q)) for q in queries])
bm25_rank = np.argsort(-bm25_scores, axis=1)[:, :TOP_N]
record("bm25", bm25_rank)

# %% [markdown]
# ## 8. Model Utama
#
# **Dense retrieval** dengan e5, yang bisa menemukan dokumen walau katanya berbeda. **Hybrid (RRF)** menggabungkan
# peringkat BM25 dan dense tanpa parameter yang perlu difit.

# %%
def rrf_merge(rank_a: np.ndarray, rank_b: np.ndarray, n_docs: int) -> np.ndarray:
    """Reciprocal Rank Fusion: skor dijumlahkan dari kedua peringkat."""
    merged = []
    for row_a, row_b in zip(rank_a, rank_b):
        scores: dict[int, float] = {}
        for row in (row_a, row_b):
            for position, doc in enumerate(row):
                scores[int(doc)] = scores.get(int(doc), 0.0) + 1 / (RRF_K + position + 1)
        merged.append(sorted(scores, key=scores.get, reverse=True)[:TOP_N])
    return np.array(merged)


if USE_PRETRAINED:
    with stage("dense"):
        doc_vectors = e5_embed(corpus_text, prefix="passage: ")
        query_vectors = e5_embed(queries, prefix="query: ")
        dense_rank = np.argsort(-(query_vectors @ doc_vectors.T), axis=1)[:, :TOP_N]
    record("dense_e5", dense_rank)
    hybrid_rank = rrf_merge(bm25_rank, dense_rank, len(data))
    record("hybrid_rrf", hybrid_rank)
else:
    hybrid_rank = bm25_rank

# %% [markdown]
# ## 9. Post-processing
#
# **Rerank.** Dokumen teratas hasil hybrid dinilai ulang oleh cross-encoder yang membaca kueri dan dokumen
# bersamaan. Lebih akurat, tapi terlalu lambat untuk seluruh korpus, jadi hanya dipakai untuk kandidat teratas.

# %%
from transformers import AutoModelForSequenceClassification as RerankModel  # noqa: E402
from transformers import AutoTokenizer as RerankTokenizer  # noqa: E402


@torch.inference_mode()
def rerank(rankings: np.ndarray, top: int, batch_size: int = 64) -> np.ndarray:
    tokenizer = RerankTokenizer.from_pretrained(MODEL_DIR / "reranker")
    model = RerankModel.from_pretrained(MODEL_DIR / "reranker").to(DEVICE).eval()
    output = []
    for query, row in zip(queries, rankings):
        candidates = row[:top]
        scores = []
        for start in range(0, len(candidates), batch_size):
            docs = [corpus_text[int(i)] for i in candidates[start : start + batch_size]]
            tokens = tokenizer([query] * len(docs), docs, padding=True, truncation=True, max_length=256,
                               return_tensors="pt").to(DEVICE)
            scores.append(model(**tokens).logits.float().squeeze(-1).cpu().numpy())
        order = np.argsort(-np.concatenate(scores))
        output.append(np.concatenate([candidates[order], row[top:]]))
    del model
    torch.cuda.empty_cache()
    return np.array(output)


if USE_PRETRAINED:
    with stage("rerank"):
        final_rank = rerank(hybrid_rank, SETTINGS["rerank_top"])
    record("hybrid_plus_rerank", final_rank)
else:
    final_rank = hybrid_rank

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Di mana rerank dibatasi oleh kandidat?** Kalau jawaban benar tidak masuk kandidat, rerank tidak mungkin
# memperbaikinya. Nilai ini menjadi batas atas.

# %%
ceiling = np.mean([answer in row[: SETTINGS["rerank_top"]] for row, answer in zip(hybrid_rank, gold)])
print(f"jawaban ada di {SETTINGS['rerank_top']} kandidat teratas: {ceiling:.1%}")

# %% [markdown]
# **Skor per kategori** dan contoh kueri yang gagal.

# %%
position = np.array([np.where(row == a)[0][0] if a in row else TOP_N for row, a in zip(final_rank, gold)])
by_category = pd.DataFrame({"kategori": data["category"].to_numpy()[gold], "kena@10": position < 10})
print(by_category.groupby("kategori")["kena@10"].agg(["size", "mean"]).round(3))
failures = np.argsort(-position)[:5]
pd.DataFrame({"kueri": [queries[i] for i in failures],
              "teratas": [corpus_text[int(final_rank[i, 0])][:70] for i in failures],
              "seharusnya": [corpus_text[int(gold[i])][:70] for i in failures]})

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id_kueri": range(len(queries)), "dokumen_top10": [" ".join(map(str, r[:10])) for r in final_rank]}).to_csv(
    RESULT_DIR / "submission.csv", index=False
)
run_summary("retrieval", len(data), {r["metode"]: {"mrr@10": r["mrr@10"], "recall@10": r["recall@10"]} for r in results},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Hybrid + rerank terbaik: MRR@10 0,947, sedangkan BM25 0,9045 dan dense 0,7331.
# - Hybrid saja justru lebih rendah dari BM25 (0,8617). Yang menaikkan skor adalah reranker.
# - Rerank memakan 288 detik, sedangkan BM25 hanya 8 detik. Mode penuh membaca 50 kandidat, dan jawaban benar ada di kandidat 98,9%.
# - Metode tanpa model (BM25) sudah kuat, jadi aman kalau pretrained dilarang.
