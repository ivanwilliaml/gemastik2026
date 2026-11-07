# %% [markdown]
# # 05 Ekstraksi Kata Kunci Tanpa Label
#
# Mengambil kata atau frasa penting tanpa data latih. Metode dinilai pada KEPS yang punya jawaban, lalu
# diterapkan ke komentar MBG.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Mengambil kata kunci tanpa label |
# | **Unit prediksi** | Satu dokumen mendapat beberapa frasa teratas |
# | **Metrik** | Precision, recall, dan F1 pada KEPS, dengan pencocokan persis |
# | **Masalah terbesar** | Rata-rata hanya 4 frasa per dokumen dan pencocokannya harus persis, jadi skor absolut memang rendah. Yang bermakna adalah perbandingan antar metode. |
# | **Data contoh** | KEPS untuk uji, komentar MBG untuk demo |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** soal meminta kata kunci, isu utama, atau frasa penting, dan tidak ada label. Kalau ada label
# frasa kunci, pakai folder `18`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_target` dan `TEXT_COL`.
# - `DOMAIN_STOPWORDS`.
# - Bagian evaluasi KEPS boleh dibiarkan, karena itu acuan pemilihan metode untuk bahasa Indonesia.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya TF-IDF dan YAKE, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
TOP_K = 5
TEXT_COL = "Comment"
DOMAIN_STOPWORDS = {"mbg", "makan", "bergizi", "gratis", "program"}
MMR_DIVERSITY = 0.5

SETTINGS = {"cepat": {"eval_docs": 400}, "penuh": {"eval_docs": 1000}}[MODE]

seed_everything(SEED)
install_offline("yake")
print(SETTINGS)

# %% include text_utils

# %% include stopwords

# %% include embedding

# %% [markdown]
# ## 4. EDA
#
# Muat KEPS sebagai acuan, dan komentar MBG sebagai target. KEPS diberi label frasa dengan pola BIO.

# %%
def read_keps(split: str) -> list[tuple[list[str], list[str]]]:
    path = EXTRA_DATA_DIR / "indonlu/dataset/keps_keyword-extraction-prosa" / f"{split}_preprocess.txt"
    sentences, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if tokens:
                sentences.append((tokens, tags))
            tokens, tags = [], []
            continue
        word, tag = line.split("\t")
        tokens.append(word)
        tags.append(tag)
    return sentences + ([(tokens, tags)] if tokens else [])


def gold_phrases(tokens: list[str], tags: list[str]) -> set[str]:
    phrases, current = set(), []
    for word, tag in zip(tokens, tags):
        if tag == "B":
            if current:
                phrases.add(" ".join(current).lower())
            current = [word]
        elif tag == "I" and current:
            current.append(word)
        else:
            if current:
                phrases.add(" ".join(current).lower())
            current = []
    if current:
        phrases.add(" ".join(current).lower())
    return phrases


keps = (read_keps("train") + read_keps("valid") + read_keps("test"))[: SETTINGS["eval_docs"]]
eval_texts = [" ".join(tokens).lower() for tokens, _ in keps]
eval_gold = [gold_phrases(tokens, tags) for tokens, tags in keps]
target = pd.read_csv(find_file("comment_data.csv")).dropna(subset=[TEXT_COL]).head(N_ROWS)
print(f"dokumen KEPS={len(keps)} | komentar={len(target)}")
print("contoh frasa KEPS:", list(eval_gold[0])[:6])
print("rata-rata frasa per dokumen:", round(np.mean([len(g) for g in eval_gold]), 2))
text_profile(target[TEXT_COL].astype(str))

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Frasa KEPS banyak yang satu kata, misalnya `ribet` | Kandidat dibuat dari n-gram 1 sampai 3 kata |
# | Frasa teratas sering tumpang tindih | Coba buang tumpang tindih dan MMR, lalu ukur efeknya |
# | Komentar MBG punya median 17 kata dan banyak kata umum | Kandidat tidak boleh diawali atau diakhiri stopword |

# %% [markdown]
# ## 5. Preprocessing
#
# Kandidat frasa berupa n-gram 1 sampai 3 kata yang tidak diawali atau diakhiri stopword. Frasa diambil dari
# teks huruf kecil, dan tanda baca memutus frasa.

# %%
import html  # noqa: E402

all_stopwords = STOPWORDS | DOMAIN_STOPWORDS
HTML_RE = re.compile(r"<[^>]+>")
target["clean"] = target[TEXT_COL].astype(str).map(lambda t: clean_text(HTML_RE.sub(" ", html.unescape(t))))


def candidate_phrases(text: str, max_words: int = 3) -> list[str]:
    """Ambil n-gram yang tepinya bukan stopword, tanpa melewati tanda baca."""
    found = []
    for chunk in re.split(r"[^\w\s-]+", text):
        words = chunk.split()
        for n in range(1, max_words + 1):
            for i in range(len(words) - n + 1):
                gram = words[i : i + n]
                if gram[0] in all_stopwords or gram[-1] in all_stopwords or len(gram[0]) < 3:
                    continue
                found.append(" ".join(gram))
    return list(dict.fromkeys(found))


print(candidate_phrases("program mbg di sekolah kami belum berjalan dengan baik, dapur masih kotor")[:8])

# %% [markdown]
# ## 6. Validasi
#
# Metrik: precision, recall, dan F1 pada `TOP_K` frasa teratas, dengan pencocokan persis setelah huruf kecil.
# Pencocokan persis cukup keras, jadi nilai absolutnya wajar rendah. Yang penting perbandingan antar metode.

# %%
def f1_at_k(predictions: list[list[str]], golds: list[set[str]], k: int = TOP_K) -> dict[str, float]:
    tp = pred_total = gold_total = 0
    for pred, gold in zip(predictions, golds):
        top = pred[:k]
        tp += len(set(top) & gold)
        pred_total += len(top)
        gold_total += len(gold)
    precision = tp / max(pred_total, 1)
    recall = tp / max(gold_total, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4)}


assert f1_at_k([["a", "b"]], [{"a"}], 2) == {"precision": 0.5, "recall": 1.0, "f1": 0.6667}
results: list[dict] = []


def record(name: str, predictions: list[list[str]]) -> None:
    results.append({"metode": name, **f1_at_k(predictions, eval_gold)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# TF-IDF: frasa yang sering di dokumen ini tapi jarang di dokumen lain. YAKE: fitur statistik satu dokumen
# saja, tidak butuh korpus. Keduanya tanpa model dari luar.

# %%
import yake  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer as PhraseTfidf  # noqa: E402

with stage("baseline"):
    phrase_tfidf = PhraseTfidf(ngram_range=(1, 3), min_df=1, sublinear_tf=True, token_pattern=r"(?u)\b[\w-]+\b")
    corpus_matrix = phrase_tfidf.fit_transform(eval_texts)
    idf = dict(zip(phrase_tfidf.get_feature_names_out(), phrase_tfidf.idf_))

    def tfidf_rank(text: str) -> list[str]:
        candidates = candidate_phrases(text)
        return sorted(candidates, key=lambda c: -text.count(c) * idf.get(c, 0) * len(c.split()))

    tfidf_pred = [tfidf_rank(t) for t in eval_texts]
    extractor = yake.KeywordExtractor(lan="id", n=3, top=TOP_K * 3, stopwords=all_stopwords)
    yake_pred = [[kw.lower() for kw, _ in extractor.extract_keywords(t)] for t in eval_texts]
record("tfidf", tfidf_pred)
record("yake", yake_pred)

# %% [markdown]
# ## 8. Model Utama
#
# **Embedding (gaya KeyBERT).** Frasa yang embedding-nya paling dekat dengan embedding dokumen dianggap paling
# mewakili isi.

# %%
def embedding_rank_all(texts: list[str]) -> tuple[list[list[str]], list[np.ndarray], list[list[str]]]:
    """Kembalikan peringkat frasa, skor, dan kandidat untuk tiap dokumen."""
    doc_vectors = e5_embed(texts, prefix="passage: ")
    candidates = [candidate_phrases(t)[:150] for t in texts]
    unique = sorted({c for row in candidates for c in row})
    lookup = dict(zip(unique, e5_embed(unique, prefix="query: ")))
    ranked, stored = [], []
    for doc_vector, row in zip(doc_vectors, candidates):
        if not row:
            ranked.append([])
            stored.append(np.zeros((0, doc_vectors.shape[1])))
            continue
        vectors = np.vstack([lookup[c] for c in row])
        ranked.append([row[i] for i in np.argsort(-(vectors @ doc_vector))])
        stored.append(vectors)
    return ranked, [doc_vectors, stored], candidates


with stage("embedding"):
    if USE_PRETRAINED:
        embed_pred, embed_store, embed_candidates = embedding_rank_all(eval_texts)
    else:
        embed_pred = tfidf_pred
record("embedding", embed_pred)

# %% [markdown]
# ## 9. Post-processing
#
# Dua langkah: buang frasa yang terkandung di frasa lain yang peringkatnya lebih tinggi, dan MMR supaya hasilnya
# beragam, tidak lima variasi dari frasa yang sama.

# %%
def drop_overlaps(phrases: list[str]) -> list[str]:
    kept: list[str] = []
    for phrase in phrases:
        if not any(phrase in other or other in phrase for other in kept):
            kept.append(phrase)
    return kept


def mmr_rank(doc_vector, vectors, row, k=TOP_K) -> list[str]:
    """Maximal Marginal Relevance: dekat dengan dokumen, jauh dari frasa terpilih."""
    relevance = vectors @ doc_vector
    chosen = [int(np.argmax(relevance))]
    while len(chosen) < min(k, len(row)):
        redundancy = (vectors @ vectors[chosen].T).max(1)
        score = (1 - MMR_DIVERSITY) * relevance - MMR_DIVERSITY * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [row[i] for i in chosen]


record("tfidf_buang_tumpang_tindih", [drop_overlaps(p) for p in tfidf_pred])
record("yake_buang_tumpang_tindih", [drop_overlaps(p) for p in yake_pred])
if USE_PRETRAINED:
    doc_vectors, stored = embed_store
    mmr_pred = [mmr_rank(d, v, c) if c else [] for d, v, c in zip(doc_vectors, stored, embed_candidates)]
    record("embedding_mmr", mmr_pred)

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results).sort_values("f1", ascending=False)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Contoh hasil** dibanding jawaban, untuk menilai apakah frasa yang "salah" sebenarnya masih masuk akal.

# %%
best_method = result_table.iloc[0]["metode"]
named = {"tfidf": tfidf_pred, "yake": yake_pred, "embedding": embed_pred,
         "tfidf_buang_tumpang_tindih": [drop_overlaps(p) for p in tfidf_pred],
         "yake_buang_tumpang_tindih": [drop_overlaps(p) for p in yake_pred]}
if USE_PRETRAINED:
    named["embedding_mmr"] = mmr_pred
best_pred = named[best_method]
pd.DataFrame(
    [{"teks": eval_texts[i][:80], "jawaban": sorted(eval_gold[i])[:4], "prediksi": best_pred[i][:4]} for i in range(5)]
)

# %% [markdown]
# **Penerapan ke komentar MBG.** Frasa diberi skor TF-IDF di seluruh korpus, lalu dijumlahkan, sehingga yang
# muncul adalah isu utama seluruh data dan bukan kata kunci satu komentar.

# %%
words_only = target["clean"].map(lambda t: " ".join(w for w in re.findall(r"[a-z]{3,}", t)))
corpus_tfidf = PhraseTfidf(ngram_range=(1, 3), min_df=5, max_df=0.2, sublinear_tf=True, token_pattern=r"\b[a-z]{3,}\b")
matrix = corpus_tfidf.fit_transform(words_only)
scores = np.asarray(matrix.sum(0)).ravel()
order = np.argsort(-scores)
names = corpus_tfidf.get_feature_names_out()
mbg_phrases = [names[i] for i in order if not (set(names[i].split()) & all_stopwords)][:400]
mbg_top = drop_overlaps(mbg_phrases)[:25]
pd.DataFrame({"frasa": mbg_top}).to_csv(RESULT_DIR / "isu_mbg.csv", index=False)
print(", ".join(mbg_top))

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
submission = pd.DataFrame({"id": range(len(best_pred)), "kata_kunci": ["; ".join(p[:TOP_K]) for p in best_pred]})
submission.to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("kata_kunci", len(target), {r["metode"]: r["f1"] for r in results}, needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Embedding + MMR terbaik: F1 0,2706, sedangkan YAKE 0,143 dan TF-IDF 0,0971.
# - MMR menaikkan embedding dari 0,1642 ke 0,2706. Membuang tumpang tindih saja justru menurunkan YAKE dan TF-IDF.
# - Tanpa label, tetap pakai metode yang terbukti di KEPS, lalu tunjukkan hasilnya pada data soal.
# - Kata kunci MBG yang muncul: korupsi, rakyat, anak, sekolah, dapur, harga, SPPG, petani.
