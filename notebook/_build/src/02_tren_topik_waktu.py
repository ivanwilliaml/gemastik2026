# %% [markdown]
# # 02 Tren Topik dan Sentimen per Waktu
#
# Bagaimana topik dan sentimen berubah dari waktu ke waktu. Contoh: komentar YouTube tentang MBG yang punya
# tanggal. Label sentimen di sini hasil zero-shot, jadi trennya dibaca sebagai indikasi arah.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Melihat perubahan topik dan sentimen dari waktu ke waktu |
# | **Unit prediksi** | Satu komentar mendapat topik dan polaritas, lalu dirata-rata per minggu |
# | **Metrik** | NPMI paruh awal dan akhir, titik perubahan |
# | **Masalah terbesar** | Topik bergeser seiring waktu: NPMI turun dari -0,0155 di paruh awal ke -0,1174 di paruh akhir pada k=6, jadi satu model topik tidak menjelaskan seluruh periode. |
# | **Data contoh** | 5.000 komentar bertanggal, 17 Februari sampai 24 Juni 2026 |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** data punya kolom waktu, dan soal meminta analisis perubahan opini atau dampak suatu
# peristiwa, misalnya sebelum dan sesudah kasus keracunan. Topik tanpa kolom waktu ada di folder `01`.
#
# **Yang perlu diganti kalau datanya beda:**
# - `load_data`, `TEXT_COL`, `TIME_COL`.
# - `N_TOPICS` dan `DOMAIN_STOPWORDS`.
# - `HYPOTHESES` kalau sentimennya berbeda.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: sentimen memakai kata kunci, tanpa file model dari luar
SEED = 42
N_ROWS = 5000
TEXT_COL, TIME_COL = "comment", "published_at"
DOMAIN_STOPWORDS = {"mbg", "makan", "bergizi", "gratis", "program"}
HYPOTHESES = {
    "negatif": "Teks ini berisi kritik, keluhan, atau penolakan.",
    "netral": "Teks ini hanya menyampaikan informasi secara netral.",
    "positif": "Teks ini berisi dukungan, pujian, atau hal yang baik.",
}
NEGATIVE_WORDS = "korupsi gagal buruk bubarkan keracunan bohong rugi parah stop tolak".split()
POSITIVE_WORDS = "bagus mantap setuju dukung semoga terima kasih hebat baik sukses".split()
MIN_ROWS_PER_PERIOD = 30

SETTINGS = {
    "cepat": {"n_topics": 8, "topic_grid": [8], "sentiment_rows": 1500},
    "penuh": {"n_topics": 8, "topic_grid": [6, 8, 10, 12], "sentiment_rows": 5000},
}[MODE]

seed_everything(SEED)
install_offline("ruptures")
print(SETTINGS)

# %% include text_utils

# %% include stopwords

# %% [markdown]
# ## 4. EDA
#
# Muat data, lalu cek rentang waktunya. Kalau rentangnya pendek atau komentar menumpuk di satu periode, tren
# tidak bisa dibaca sembarangan.

# %%
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(find_file("komentar-mbg.csv")).dropna(subset=[TEXT_COL, TIME_COL])
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="coerce", utc=True)
    frame = frame.dropna(subset=[TIME_COL]).drop_duplicates(TEXT_COL)
    return frame.sample(n=min(N_ROWS, len(frame)), random_state=SEED).sort_values(TIME_COL).reset_index(drop=True)


data = load_data()
span_days = (data[TIME_COL].max() - data[TIME_COL].min()).days
FREQ = "D" if span_days < 60 else ("W" if span_days < 400 else "MS")
data["periode"] = data[TIME_COL].dt.tz_localize(None).dt.to_period(FREQ[0]).dt.start_time
print(f"baris={len(data)} | {data[TIME_COL].min().date()} s.d. {data[TIME_COL].max().date()} | periode={FREQ}")
text_profile(data[TEXT_COL])

# %% [markdown]
# Jumlah komentar per periode.

# %%
counts_time = data.groupby("periode").size()
fig, ax = plt.subplots()
ax.plot(counts_time.index, counts_time.values, color="#4C72B0")
ax.axhline(MIN_ROWS_PER_PERIOD, color="#C44E52", ls="--", label="batas minimum")
ax.set(title="Jumlah komentar per periode", xlabel="waktu", ylabel="jumlah komentar", yscale="log")
ax.legend()
plt.tight_layout()
plt.show()
print("periode dengan cukup data:", int((counts_time >= MIN_ROWS_PER_PERIOD).sum()), "dari", len(counts_time))

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | 18 dari 19 minggu punya cukup data | Agregasi mingguan layak dipakai |
# | NPMI paruh akhir lebih rendah pada semua k yang dicoba | Pilih k dengan NPMI paruh akhir tertinggi, yaitu k=6, dan laporkan bahwa topik bergeser |
# | Banyak komentar membahas tokoh, bukan programnya | Isi topik dibaca manual sebelum disimpulkan |

# %% [markdown]
# ## 5. Preprocessing
#
# Sama seperti folder `01`: HTML dibuang, slang dibakukan untuk kata kunci, dan stopword dibuang hanya pada
# versi `words`.

# %%
import html  # noqa: E402

slang = load_slang_dict()
all_stopwords = STOPWORDS | DOMAIN_STOPWORDS
HTML_RE = re.compile(r"<[^>]+>")
data["clean"] = data[TEXT_COL].astype(str).map(lambda t: clean_text(HTML_RE.sub(" ", html.unescape(t))))
data["words"] = data["clean"].map(
    lambda t: " ".join(w for w in re.findall(r"[a-z]{3,}", normalize_slang(t, slang)) if w not in all_stopwords)
)
data = data[data["words"].str.split().str.len() >= 3].reset_index(drop=True)
print("baris setelah dibersihkan:", len(data))

# %% [markdown]
# ## 6. Validasi
#
# Validasi temporal: model dilatih dari **paruh awal**, lalu kualitas topiknya dinilai pada **paruh akhir**.
# Kalau NPMI jatuh jauh, topiknya berubah seiring waktu dan satu model topik tidak cukup.

# %%
from scipy.sparse import csr_matrix  # noqa: E402
from sklearn.decomposition import NMF  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # noqa: E402


def npmi_coherence(topics, doc_word) -> float:
    """Rata-rata NPMI pasangan kata kunci, dihitung dari kemunculan dalam dokumen."""
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


_toy = csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int32))
assert abs(npmi_coherence([[0, 1]], _toy) - 1.0) < 1e-9

vectorizer = CountVectorizer(min_df=5, max_df=0.2, token_pattern=r"\b[a-z]{3,}\b")
counts = vectorizer.fit_transform(data["words"])
vocab = np.array(vectorizer.get_feature_names_out())
presence = (counts > 0).astype(np.int32)
tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
half = len(data) // 2


def top_words(components) -> list[list[int]]:
    return [list(np.argsort(-row)[:10]) for row in components]


results: list[dict] = []
for k in SETTINGS["topic_grid"]:
    early = NMF(k, random_state=SEED, init="nndsvda", max_iter=400).fit(tfidf[:half])
    topics = top_words(early.components_)
    results.append({"k": k, "npmi_paruh_awal": round(npmi_coherence(topics, presence[:half]), 4),
                    "npmi_paruh_akhir": round(npmi_coherence(topics, presence[half:]), 4)})
pd.DataFrame(results)

# %% [markdown]
# ## 7. Baseline
#
# Tren frekuensi kata. Sederhana, tapi sering sudah memperlihatkan peristiwa besar.

# %%
watch_words = [w for w in ["korupsi", "keracunan", "anggaran", "gizi", "dapur"] if w in vocab]
word_share = pd.DataFrame(
    {w: data["words"].str.contains(rf"\b{w}\b").groupby(data["periode"]).mean() for w in watch_words}
)
enough = counts_time[counts_time >= MIN_ROWS_PER_PERIOD].index
word_share = word_share.loc[word_share.index.isin(enough)]
fig, ax = plt.subplots()
word_share.plot(ax=ax)
ax.set(title="Porsi komentar yang menyebut kata", xlabel="waktu", ylabel="porsi komentar")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Model Utama
#
# **Topik:** NMF dipilih karena di folder `01` kata kuncinya paling koheren, dan bisa jalan tanpa model dari
# luar. **Sentimen:** zero-shot pada sampel, kalau pretrained boleh dipakai.

# %%
best_k = max(results, key=lambda r: r["npmi_paruh_akhir"])["k"]
nmf = NMF(best_k, random_state=SEED, init="nndsvda", max_iter=400)
doc_topic = nmf.fit_transform(tfidf)
data["topik"] = doc_topic.argmax(1)
topic_words = {t: ", ".join(vocab[w][:6]) for t, w in enumerate(top_words(nmf.components_))}
pd.Series(topic_words, name="kata_kunci")

# %% [markdown]
# Skor sentimen. Tanpa pretrained, sentimen dihitung dari selisih kata kunci positif dan negatif.

# %%
from transformers import AutoModelForSequenceClassification as NliModel  # noqa: E402
from transformers import AutoTokenizer as NliTokenizer  # noqa: E402


@torch.inference_mode()
def zero_shot(texts, hypotheses, batch_size=64) -> np.ndarray:
    tokenizer = NliTokenizer.from_pretrained(MODEL_DIR / "mdeberta_nli")
    model = NliModel.from_pretrained(MODEL_DIR / "mdeberta_nli").to(DEVICE).eval()
    entail = model.config.label2id.get("entailment", 0)
    scores = np.zeros((len(texts), len(hypotheses)))
    for h_index, hypothesis in enumerate(hypotheses):
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(batch, [hypothesis] * len(batch), padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                scores[start : start + batch_size, h_index] = model(**tokens).logits.float()[:, entail].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return torch.softmax(torch.tensor(scores), dim=1).numpy()


sample = data.sample(n=min(SETTINGS["sentiment_rows"], len(data)), random_state=SEED)
with stage("sentimen"):
    if USE_PRETRAINED:
        sentiment_probs = zero_shot(sample["clean"].tolist(), list(HYPOTHESES.values()))
        polarity = sentiment_probs[:, 2] - sentiment_probs[:, 0]
    else:
        negative_hits = sample["clean"].str.count("|".join(NEGATIVE_WORDS))
        positive_hits = sample["clean"].str.count("|".join(POSITIVE_WORDS))
        polarity = np.sign(positive_hits - negative_hits).to_numpy(dtype=float)
data.loc[sample.index, "polaritas"] = polarity
print(f"polaritas rata-rata: {np.mean(polarity):+.3f}")

# %% [markdown]
# ## 9. Post-processing
#
# Periode dengan data terlalu sedikit dibuang, tren dihaluskan dengan rata-rata bergerak, lalu titik perubahan
# dicari dengan `ruptures`. Titik perubahan adalah petunjuk untuk dicocokkan dengan peristiwa, bukan bukti
# sebab akibat.

# %%
import ruptures  # noqa: E402

topic_share = pd.crosstab(data["periode"], data["topik"], normalize="index")
topic_share = topic_share.loc[topic_share.index.isin(enough)]
smooth_topics = topic_share.rolling(3, min_periods=1, center=True).mean()
polarity_trend = data.loc[data["periode"].isin(enough)].groupby("periode")["polaritas"].mean()
smooth_polarity = polarity_trend.rolling(3, min_periods=1, center=True).mean()

change_points = []
if len(smooth_polarity.dropna()) >= 8:
    signal = smooth_polarity.dropna().to_numpy()
    detector = ruptures.Pelt(model="rbf", min_size=3).fit(signal)
    change_points = [smooth_polarity.dropna().index[i] for i in detector.predict(pen=3)[:-1]]
print("titik perubahan polaritas:", [str(p.date()) for p in change_points] or "tidak ada")

# %% [markdown]
# ## 10. Post-analysis
#
# **Porsi topik dari waktu ke waktu.**

# %%
fig, ax = plt.subplots(figsize=(9, 3.8))
smooth_topics.plot.area(ax=ax, stacked=True, alpha=0.8, linewidth=0)
ax.set(title="Porsi topik per periode", xlabel="waktu", ylabel="porsi komentar")
ax.legend(title="topik", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# **Tren polaritas** dengan titik perubahannya.

# %%
fig, ax = plt.subplots()
ax.plot(smooth_polarity.index, smooth_polarity.values, color="#4C72B0")
for point in change_points:
    ax.axvline(point, color="#C44E52", ls="--")
ax.axhline(0, color="gray", lw=0.8)
ax.set(title="Polaritas rata-rata per periode", xlabel="waktu", ylabel="positif dikurangi negatif")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Ringkasan per topik:** ukuran, polaritas, dan arah perubahan porsi dari paruh awal ke paruh akhir.

# %%
middle = data["periode"].sort_values().iloc[len(data) // 2]
summary = pd.DataFrame(
    {"kata_kunci": topic_words,
     "jumlah": data["topik"].value_counts(),
     "polaritas": data.groupby("topik")["polaritas"].mean().round(3),
     "porsi_awal": data.loc[data["periode"] < middle, "topik"].value_counts(normalize=True).round(3),
     "porsi_akhir": data.loc[data["periode"] >= middle, "topik"].value_counts(normalize=True).round(3)}
).sort_values("jumlah", ascending=False)
summary.to_csv(RESULT_DIR / "ringkasan_topik.csv")
summary

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
data[["periode", "topik", "polaritas"]].rename_axis("id").to_csv(RESULT_DIR / "submission.csv")
run_summary("tren_topik_waktu", len(data),
            {"k": int(best_k), "validasi_temporal": results, "titik_perubahan": [str(p.date()) for p in change_points]},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - Ditemukan satu titik perubahan polaritas di sekitar 23 Maret 2026.
# - Polaritas rata-rata -0,290. Topik presiden paling negatif (-0,505), dan porsinya turun dari 30,9% ke 20,2%.
# - NPMI negatif menandakan koherensi kata kunci lemah, jadi kesimpulan topik di notebook ini perlu didukung contoh komentar.
# - Polaritas berasal dari model, bukan label manusia. Sebutkan ini saat presentasi.
