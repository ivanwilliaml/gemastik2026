# %% [markdown]
# # 08 Klasifikasi Semi-Supervised
#
# Label sedikit, data tanpa label banyak. **Cara pakai: ubah sel Config, lalu Run All.**
#
# 1. Kartu Masalah
# 2. Kapan Pakai
# 3. Config
# 4. EDA (14 pemeriksaan, masing-masing berakhir pada keputusan)
# 5. Preprocessing (langkah dipilih dari temuan, lalu diuji ablasi)
# 6. Validasi
# 7. Baseline
# 8. Model Utama
# 9. Post-processing
# 10. Post-analysis
# 11. Keluaran

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Klasifikasi teks dengan sedikit label dan banyak data tanpa label |
# | **Unit prediksi** | Satu teks mendapat satu kelas |
# | **Metrik** | Sesuai soal, diatur di `METRIC` |
# | **Masalah terbesar** | Dihitung dari data, lihat pemeriksaan 4.14 |

# %% [markdown]
# ## 2. Kapan Pakai
#
# **Pakai kalau** soal memberi data berlabel dan tanpa label, atau semuanya berlabel. Biner dan multi-kelas sama saja.
#
# | Kondisi data | Yang terjadi |
# |---|---|
# | Sebagian label kosong | Baris kosong menjadi data tanpa label, dan self-training dicoba |
# | Semua berlabel | Self-training dilewati |
# | Label test ada atau tidak | Kolom skor test muncul hanya kalau labelnya ada, dan tidak pernah dipakai untuk memilih |
# | Test bergeser dari data latih | Skor diberi bobot mirip test, dan pengali kelas dimatikan |
# | Waktu mepet | Uji setelan, student, dan seed tambahan dilewati otomatis |
# | Pretrained dilarang atau file model tidak ada | Otomatis memakai TF-IDF |
#
# **Copot pasang.** Semua keputusan disimpan di `settings`, dan nilai bawaannya aman. Jadi sel EDA, uji ablasi, atau
# post-processing boleh dihapus tanpa merusak sel berikutnya. Hanya sel yang bertanda **wajib** yang harus tetap ada.
# Mesin pelatihan ada di folder `nlp_kit`, sedangkan analisis dan keputusannya ada di notebook ini supaya bisa diubah.

# %% [markdown]
# ## 3. Config
#
# Hanya sel ini yang perlu diubah. `TEXT_COL` boleh berupa daftar kolom, misalnya `["judul", "isi"]`.

# %%
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TEXT_COL = "text"
LABEL_COL = "label"
ID_COL = None  # nama kolom id di test, None kalau tidak ada
GROUP_COL = None  # kolom yang tidak boleh lintas fold, misalnya id video
POSITIVE_CLASS = None  # hanya untuk METRIC "f1"
METRIC = "macro_f1"  # macro_f1, f1, weighted_f1, atau accuracy
WAKTU_MENIT = 60  # jumlah fold, epoch, dan model menyesuaikan batas ini
MODEL_DIR = "../../offline/models"
MODELS = ["indobertweet", "indobert"]  # kosongkan untuk hanya memakai TF-IDF
PAKAI_TEST_SAAT_LATIH = False  # True hanya kalau aturan lomba membolehkan
SEED = 42

# %% [markdown]
# **Wajib.** Muat `nlp_kit` dan siapkan tempat menyimpan temuan (`notes`) serta keputusan (`settings`). Mode offline
# sudah aktif dari sini.

# %%
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import LabelSpreading

sys.path.insert(0, "../..")
import nlp_kit as nk

clock = nk.start(WAKTU_MENIT, SEED)
notes = nk.Notes()
settings = nk.Settings()
usable_models = nk.available_models(MODELS, MODEL_DIR)
score = nk.get_metric(METRIC)

# %% [markdown]
# ## 4. EDA
#
# Tiap pemeriksaan menjawab satu pertanyaan, dan menghasilkan temuan lalu keputusan untuk langkah berikutnya.
#
# ### 4.1 Kolom apa saja yang ada?
#
# Kolom lain bisa menjadi fitur, kunci grup, atau justru bocoran label. Kolom yang ada di latih tetapi tidak ada di test
# tidak boleh dipakai sebagai fitur.

# %%
raw_train, raw_test = nk.load_table(TRAIN_PATH), nk.load_table(TEST_PATH)


def column_audit(frame):
    example = [str(frame[c].dropna().iloc[0])[:40] if frame[c].notna().any() else "" for c in frame.columns]
    return pd.DataFrame({"tipe": frame.dtypes.astype(str), "kosong_%": (100 * frame.isna().mean()).round(1),
                         "unik": frame.nunique(), "contoh": example})


text_cols = [TEXT_COL] if isinstance(TEXT_COL, str) else list(TEXT_COL)
known = {*text_cols, LABEL_COL, ID_COL, GROUP_COL, "label_audit"}
extra_cols = [c for c in raw_train.columns if c not in known]
display(column_audit(raw_train))
filled = raw_train[LABEL_COL].notna()
for column in extra_cols:
    values = raw_train.loc[filled, column].astype(str)
    if 2 <= values.nunique() <= 50:
        table = pd.crosstab(values, raw_train.loc[filled, LABEL_COL])
        strength = np.sqrt(chi2_contingency(table)[0] / (table.to_numpy().sum() * (min(table.shape) - 1)))
        if strength >= 0.5:
            notes.add(f"Kolom {column} sangat terkait label, Cramer's V {strength:.2f}",
                      "Periksa dulu, ini bisa bocoran. Jangan dipakai tanpa alasan", 2)
    if column not in raw_test.columns:
        notes.add(f"Kolom {column} tidak ada di test", "Tidak boleh dipakai sebagai fitur")
    if re.search("tanggal|date|time|waktu", column, re.I) and column in raw_test.columns:
        latih, uji = pd.to_datetime(raw_train[column], errors="coerce"), pd.to_datetime(raw_test[column], errors="coerce")
        if latih.notna().any() and uji.notna().any():
            notes.add(f"Kolom {column}: latih {latih.min():%Y-%m-%d} s.d. {latih.max():%Y-%m-%d}, test {uji.min():%Y-%m-%d} s.d. {uji.max():%Y-%m-%d}",
                      "Kalau test lebih baru, pertimbangkan validasi berbasis waktu", 1)
if extra_cols:
    notes.add(f"Ada {len(extra_cols)} kolom di luar teks dan label: {extra_cols[:6]}", "Notebook ini hanya memakai teks. Tambahkan kolom itu kalau sinyalnya jelas")

# %% [markdown]
# ### 4.2 Datanya utuh? **(wajib)**
#
# Memuat data ke bentuk baku. Kolom `clean` hanya membuang gangguan teknis, dan dipakai semua pemeriksaan berikutnya.
# `represent` mengubah teks menjadi vektor: embedding kalau ada model, TF-IDF + SVD kalau tidak.

# %%
data = nk.load_task(TRAIN_PATH, TEST_PATH, TEXT_COL, LABEL_COL, ID_COL, GROUP_COL, POSITIVE_CLASS)
plan = nk.plan_training(WAKTU_MENIT, len(data.labeled))
n_classes = len(data.classes)
for frame in (data.labeled, data.unlabeled, data.test):
    frame["clean"] = frame["text"].map(nk.clean_text)
all_clean = data.labeled["clean"].tolist() + data.unlabeled["clean"].tolist() + data.test["clean"].tolist()
sparse_pipeline = None


def represent(texts):
    global sparse_pipeline
    if usable_models:
        return nk.embed_texts(Path(MODEL_DIR) / usable_models[0], list(texts), settings.train.max_len)
    if sparse_pipeline is None:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        sparse_pipeline = make_pipeline(tfidf, TruncatedSVD(min(100, len(all_clean) // 4), random_state=SEED)).fit(all_clean)
    return sparse_pipeline.transform(list(texts))


checks = {
    "baris latih berlabel": len(data.labeled),
    "baris latih tanpa label": len(data.unlabeled),
    "baris latih dibuang karena teks kosong": len(raw_train) - data.n_rows,
    "baris test": len(data.test),
    "id test duplikat": int(data.test["id"].duplicated().sum()),
    "test kosong setelah dibersihkan": int((data.test["clean"] == "").sum()),
}
if data.has_test_labels:
    checks["kelas test yang tidak ada di latih"] = len(set(data.test["label"].dropna()) - set(data.classes))
display(pd.Series(checks, name="jumlah").to_frame())
if checks["id test duplikat"]:
    notes.add(f"{checks['id test duplikat']} id test duplikat", "Tanyakan ke panitia sebelum submit", 2)
if checks["test kosong setelah dibersihkan"]:
    notes.add(f"{checks['test kosong setelah dibersihkan']} teks test kosong setelah dibersihkan", "Tetap diprediksi, karena jumlah baris submission harus utuh", 1)
data.labeled[["text", "label"]].head(3)

# %% [markdown]
# ### 4.3 Seberapa timpang labelnya?
#
# Pada macro-F1, kelas kecil bernilai sama dengan kelas besar. Kelas dengan baris sangat sedikit membuat skor per fold
# tidak stabil.

# %%
counts = data.labeled["label"].value_counts().reindex(data.classes)
shares = counts / counts.sum()
display(pd.DataFrame({"baris": counts, "proporsi": shares.round(3)}))
counts.plot.barh(color="#4C72B0", title="Distribusi kelas pada data berlabel", xlabel="jumlah baris", ylabel="kelas")
plt.tight_layout()
plt.show()
ratio = counts.max() / max(counts.min(), 1)
if ratio >= 3:
    rare = ", ".join(f"{name} {share:.1%}" for name, share in shares.nsmallest(2).items())
    notes.add(f"Kelas timpang, rasio {ratio:.1f} kali: {rare}", "Bobot kelas dipakai, kuota pseudo-label per kelas, pengali kelas dicoba", 2)
else:
    notes.add(f"Kelas cukup seimbang, rasio {ratio:.1f} kali", "Bobot kelas diuji di 8.1, tidak diasumsikan membantu")
if (counts < plan.folds).any():
    notes.add(f"Kelas {list(counts[counts < plan.folds].index)} kurang dari {plan.folds} baris", "Skor kelas ini sangat tidak stabil, jangan disimpulkan apa-apa darinya", 2)
if len(data.labeled) < 1000:
    notes.add(f"Label hanya {len(data.labeled)} baris", "Selisih skor kecil dianggap noise. Model baru dipakai kalau peluang lebih baiknya minimal 0,9", 2)

# %% [markdown]
# ### 4.4 Baris yang diberi label dipilih secara acak?
#
# Kalau baris berlabel mudah dibedakan dari baris tanpa label, pemberian labelnya tidak acak. Akibatnya presisi
# pseudo-label yang dihitung dari data berlabel belum tentu berlaku di data tanpa label.

# %%
if len(data.unlabeled):
    bias_auc, _ = nk.adversarial_scores(data.labeled["clean"].tolist(), data.unlabeled["clean"].tolist())
    if bias_auc >= 0.65:
        settings.target_precision = 0.95
        notes.add(f"Baris berlabel tidak acak, AUC {bias_auc:.3f}", "Ambang presisi pseudo-label dinaikkan ke 0,95", 2)
    else:
        notes.add(f"Baris berlabel terlihat acak, AUC {bias_auc:.3f}", "Presisi dari data berlabel bisa dipakai untuk memilih pseudo-label")
else:
    notes.add("Semua baris berlabel", "Self-training otomatis dilewati")

# %% [markdown]
# ### 4.5 Seberapa panjang teksnya, dan panjang token berapa yang cukup?
#
# Panjang dihitung memakai tokenizer asli kalau tersedia. Dari sini `max_len` ditentukan, supaya tidak ada waktu terbuang
# untuk padding dan tidak banyak teks terpotong. Selisih panjang antar kelas atau antar bagian juga diperiksa.

# %%
tokenizer = nk.load_tokenizer(MODELS, MODEL_DIR)
parts = {"berlabel": data.labeled, "tanpa_label": data.unlabeled, "test": data.test}
parts = {name: frame for name, frame in parts.items() if len(frame)}
word_lengths = {name: frame["clean"].str.split().str.len() for name, frame in parts.items()}
quantiles = pd.DataFrame({name: w.describe(percentiles=[0.5, 0.95, 0.99]) for name, w in word_lengths.items()}).drop("count").round(1)
display(quantiles)
display(word_lengths["berlabel"].groupby(data.labeled["label"]).describe()[["50%", "75%", "max"]].rename_axis("kelas"))

fig, axes = plt.subplots(1, 2)
for name, lengths in word_lengths.items():
    lengths.clip(upper=lengths.quantile(0.99)).plot.hist(bins=30, density=True, alpha=0.5, ax=axes[0], label=name)
axes[0].set(title="Panjang teks per bagian", xlabel="jumlah kata", ylabel="kepadatan")
axes[0].legend()
for name, group in word_lengths["berlabel"].groupby(data.labeled["label"]):
    group.clip(upper=word_lengths["berlabel"].quantile(0.99)).plot.hist(bins=25, density=True, alpha=0.5, ax=axes[1], label=name)
axes[1].set(title="Panjang teks per kelas", xlabel="jumlah kata", ylabel="kepadatan")
axes[1].legend()
plt.tight_layout()
plt.show()

if tokenizer is not None:
    token_lengths = np.array([len(ids) for ids in tokenizer(all_clean, add_special_tokens=False)["input_ids"]])
else:
    token_lengths = np.array([int(len(text.split()) * 1.6) for text in all_clean])
settings.train.max_len = int(np.clip(np.ceil((np.quantile(token_lengths, 0.98) + 2) / 32) * 32, 64, 256))
truncated = float((token_lengths + 2 > settings.train.max_len).mean())
settings.train.head_tail = truncated >= 0.02
how = "potong awal dan akhir teks" if settings.train.head_tail else "potong akhir teks"
notes.add(f"98% teks di bawah {np.quantile(token_lengths, 0.98):.0f} token, {truncated:.1%} terpotong pada {settings.train.max_len}",
          f"max_len {settings.train.max_len}, {how}", 1 if settings.train.head_tail else 0)
gap = ks_2samp(word_lengths["berlabel"], word_lengths["test"]).statistic
if gap >= 0.15:
    notes.add(f"Panjang teks test berbeda dari latih, KS {gap:.2f}", "Analisis skor per kelompok panjang di 10.4", 1)

# %% [markdown]
# ### 4.6 Pola apa yang ada di teks?
#
# Persen baris yang mengandung tiap pola, per bagian dan per kelas. Pola yang muncul menentukan langkah preprocessing
# yang layak diuji. Pola yang timpang antar kelas menandakan sinyal, jadi jangan asal dibuang.

# %%
rates = pd.DataFrame({name: nk.pattern_rates(frame["text"]) for name, frame in parts.items()}).round(1)
by_class = data.labeled.groupby("label")["text"].apply(nk.pattern_rates).unstack(0).round(1)
display(rates.join(by_class))
triggered = [step for step, pattern in nk.STEP_TRIGGERS.items() if rates.loc[pattern].max() >= 1]
settings.candidate_steps = triggered + ["slang", "tanda_baca", "stopword"]
notes.add(f"Pola yang muncul di minimal 1% baris memicu langkah {triggered or 'tidak ada'}", "Langkah itu diuji di 5.3, bukan langsung dipakai")
technical = rates.loc[["url", "mention", "html", "emoji_byte", "token_anonim"]].max(axis=1)
if technical.max() > 0:
    notes.add(f"Gangguan teknis: {technical[technical > 0].to_dict()}", "Dibersihkan di langkah wajib 5.1")
signal = (by_class.max(axis=1) - by_class.min(axis=1)).sort_values(ascending=False)
if signal.iloc[0] >= 15:
    notes.add(f"Pola {signal.index[0]} timpang antar kelas, selisih {signal.iloc[0]:.0f} poin persen", "Pola ini membawa sinyal, jadi dipertahankan untuk transformer", 1)

# %% [markdown]
# ### 4.7 Seberapa informal bahasanya?
#
# Porsi kata yang ada di kamus slang, kata yang dipecah tokenizer menjadi banyak potongan, dan porsi token yang tidak
# dikenal. Makin tinggi, makin mungkin normalisasi membantu dan model khusus media sosial lebih cocok.

# %%
slang = nk.load_slang(MODEL_DIR)
words = pd.Series(" ".join(data.labeled["clean"].tolist() + data.unlabeled["clean"].tolist()).split())
word_counts = words.value_counts()
slang_hits = word_counts[word_counts.index.isin(slang)]
slang_share = slang_hits.sum() / max(len(words), 1)
print(f"porsi kata slang: {slang_share:.1%} | contoh:", dict(slang_hits.head(12)))
if tokenizer is not None:
    common = word_counts.head(2000)
    pieces = pd.Series({word: len(tokenizer.tokenize(word)) for word in common.index})
    unknown = pd.Series({word: tokenizer.unk_token in tokenizer.tokenize(word) for word in common.index})
    print("kata umum yang paling terpecah:", pieces.sort_values(ascending=False).head(10).to_dict())
    unknown_share = common[unknown].sum() / common.sum()
    if unknown_share >= 0.02:
        notes.add(f"{unknown_share:.1%} kata umum menjadi token tidak dikenal", "Normalisasi diuji di 5.4, dan bahasa pada contoh salah diperiksa", 1)
if slang_share >= 0.05:
    notes.add(f"Bahasa informal, {slang_share:.1%} kata ada di kamus slang", "Normalisasi slang diuji pada TF-IDF dan transformer", 1)
else:
    notes.add(f"Slang sedikit, {slang_share:.1%} kata", "Normalisasi slang kemungkinan tidak banyak berpengaruh")

# %% [markdown]
# ### 4.8 Ada teks kembar?
#
# Teks kembar yang lintas fold membuat skor terlalu optimistis. Teks sama dengan label berbeda menunjukkan batas skor
# yang bisa dicapai. Teks test yang persis ada di data berlabel bisa langsung disalin labelnya kalau labelnya konsisten.

# %%
labeled_clean = data.labeled["clean"]
near_groups = nk.near_duplicate_groups(labeled_clean.tolist())
sizes = pd.Series(near_groups).value_counts()
conflict = data.labeled.groupby("clean")["label"].nunique()
conflict_texts = conflict[conflict > 1]
test_overlap = data.test["clean"].isin(set(labeled_clean)).mean()
pool_overlap = data.unlabeled["clean"].isin(set(labeled_clean)).mean() if len(data.unlabeled) else 0.0
duplicate_share = labeled_clean.duplicated().mean()
display(pd.Series({
    "duplikat persis di data berlabel": f"{duplicate_share:.1%}",
    "baris dalam grup near-duplicate": f"{sizes[sizes > 1].sum() / len(near_groups):.1%}",
    "grup terbesar": int(sizes.iloc[0]),
    "teks sama dengan label berbeda": len(conflict_texts),
    "test yang persis ada di data berlabel": f"{test_overlap:.1%}",
    "tanpa label yang persis ada di data berlabel": f"{pool_overlap:.1%}",
}, name="nilai").to_frame())
if len(conflict_texts):
    notes.add(f"{len(conflict_texts)} teks sama punya label berbeda", "Label berisik. Diselesaikan dengan suara terbanyak di 5.2", 2)
if sizes.iloc[0] > 1:
    notes.add(f"Grup near-duplicate terbesar {sizes.iloc[0]} baris", "Grup yang sama tidak boleh lintas fold", 1)
if test_overlap >= 0.01:
    settings.copy_duplicates = True
    notes.add(f"{test_overlap:.1%} teks test persis ada di data berlabel", "Salin label kalau konsisten, lihat 9.5", 1)

# %% [markdown]
# ### 4.9 Kata apa yang membedakan tiap kelas?
#
# Log-odds dengan prior Dirichlet: kata yang khas pada satu kelas dibanding kelas lain. Kalau satu kata hampir selalu
# berarti satu kelas dan sering muncul, labelnya mungkin dibuat dari leksikon. Model akan terlihat bagus karena menghafal
# kata itu.

# %%
vectorizer = CountVectorizer(token_pattern=r"\S+", min_df=2, binary=True)
matrix = vectorizer.fit_transform(labeled_clean)
vocabulary = np.array(vectorizer.get_feature_names_out())
total = np.asarray(matrix.sum(0)).ravel().astype(float)
alpha = total / total.sum() * 500
terms, shortcuts = {}, []
for index, name in enumerate(data.classes):
    inside = data.y == index
    in_class = np.asarray(matrix[inside].sum(0)).ravel()
    out_class = total - in_class
    log_odds = np.log((in_class + alpha) / (in_class.sum() + alpha.sum() - in_class - alpha)) \
        - np.log((out_class + alpha) / (out_class.sum() + alpha.sum() - out_class - alpha))
    z_score = log_odds / np.sqrt(1 / (in_class + alpha) + 1 / (out_class + alpha))
    terms[name] = vocabulary[np.argsort(-z_score)[:8]]
    strong = (in_class >= 0.2 * inside.sum()) & (in_class / np.maximum(total, 1) >= 0.95) & (in_class >= 5)
    shortcuts += [f"{word} ({name})" for word in vocabulary[strong]]
display(pd.DataFrame(terms))
if shortcuts:
    notes.add(f"Kata hampir menentukan kelas: {shortcuts[:6]}", "Label mungkin dari leksikon, atau kelas terkait topik. Skor tinggi belum tentu berlaku di test", 2)
else:
    notes.add("Tidak ada satu kata yang menentukan kelas", "Tidak ada tanda label dibuat dari leksikon sederhana")

# %% [markdown]
# ### 4.10 Test bergeser dari data latih?
#
# Adversarial validation: model dilatih membedakan latih dan test. AUC dekat 0,5 berarti sama, dan AUC tinggi berarti
# bergeser. Kata pembeda menjelaskan apa bedanya. Kalau bergeser, skor OOF biasa terlalu optimistis.

# %%
reference = data.unlabeled if len(data.unlabeled) else data.labeled
settings.shift_auc, adversarial = nk.adversarial_scores(reference["clean"].tolist(), data.test["clean"].tolist())
display(nk.adversarial_terms(reference["clean"].tolist(), data.test["clean"].tolist(), 8))
train_vocabulary = set(" ".join(reference["clean"]).split())
test_words = " ".join(data.test["clean"]).split()
unseen = np.mean([word not in train_vocabulary for word in test_words])
plt.hist(adversarial[: len(reference)], bins=30, alpha=0.6, density=True, label="latih")
plt.hist(adversarial[len(reference) :], bins=30, alpha=0.6, density=True, label="test")
plt.title(f"Skor mirip test, AUC {settings.shift_auc:.3f}")
plt.xlabel("skor mirip test")
plt.ylabel("kepadatan")
plt.legend()
plt.tight_layout()
plt.show()
print(f"kata test yang tidak ada di latih: {unseen:.1%}")
if settings.has_shift:
    notes.add(f"Test bergeser dari data latih, AUC adversarial {settings.shift_auc:.3f}", "Skor OOF diberi bobot mirip test, dan pengali kelas dimatikan", 3)
else:
    notes.add(f"Tidak ada pergeseran kuat, AUC adversarial {settings.shift_auc:.3f}", "Skor OOF bisa dipakai memilih model")

# %% [markdown]
# ### 4.11 Kelasnya terpisah, dan data tanpa label dekat dengan data berlabel?
#
# Self-training hanya membantu kalau baris yang berdekatan cenderung berkelas sama. Ini diukur dengan KNN pada vektor
# teks. Jarak ke tetangga berlabel menunjukkan apakah data tanpa label dan test sekitar data berlabel.

# %%
labeled_vectors = np.asarray(represent(labeled_clean.tolist()))
splitter = StratifiedKFold(3, shuffle=True, random_state=SEED)
knn_pred = cross_val_predict(KNeighborsClassifier(5, metric="cosine"), labeled_vectors, data.y, cv=splitter)
knn_score = score(data.y, knn_pred)
majority_score = score(data.y, np.full(len(data.y), np.bincount(data.y).argmax()))
pick = np.random.default_rng(SEED).permutation(len(data.y))[:2000]
separation = silhouette_score(labeled_vectors[pick], data.y[pick], metric="cosine") if len(set(data.y[pick])) > 1 else 0.0
test_vectors = np.asarray(represent(data.test["clean"].tolist()))
neighbors = NearestNeighbors(n_neighbors=2, metric="cosine").fit(labeled_vectors)
distances = {"berlabel": neighbors.kneighbors(labeled_vectors)[0][:, 1], "test": neighbors.kneighbors(test_vectors)[0][:, 0]}
if len(data.unlabeled):
    distances["tanpa_label"] = neighbors.kneighbors(np.asarray(represent(data.unlabeled["clean"].tolist())))[0][:, 0]
display(pd.DataFrame({name: pd.Series(d).describe()[["50%", "75%"]] for name, d in distances.items()}).round(3))
pca = PCA(2, random_state=SEED).fit(labeled_vectors)
plane = pca.transform(test_vectors)
plt.scatter(plane[:, 0], plane[:, 1], s=4, color="lightgray", label="test")
plane = pca.transform(labeled_vectors)
for index, name in enumerate(data.classes):
    plt.scatter(plane[data.y == index, 0], plane[data.y == index, 1], s=8, label=name)
plt.title(f"Peta teks, silhouette {separation:.3f}")
plt.xlabel("komponen 1")
plt.ylabel("komponen 2")
plt.legend(markerscale=2)
plt.tight_layout()
plt.show()
has_pool = len(data.unlabeled) > 0 or PAKAI_TEST_SAAT_LATIH
if knn_score >= majority_score + 0.15:
    notes.add(f"Tetangga dekat cenderung sekelas: KNN {knn_score:.3f}, mayoritas {majority_score:.3f}", "Syarat self-training terpenuhi" if has_pool else "Representasi teks cukup informatif")
else:
    notes.add(f"Kelas tidak terpisah pada vektor ini: KNN {knn_score:.3f}, mayoritas {majority_score:.3f}", "Model rumit harus lolos gerbang di 9.1, dan baseline bisa jadi yang terbaik", 2)
if np.median(distances["test"]) > 1.5 * np.median(distances["berlabel"]):
    notes.add("Test jauh dari tetangga berlabel", "Bukti tambahan pergeseran. Keputusan makin mengandalkan skor mirip test", 1)

# %% [markdown]
# ### 4.12 Seberapa berisik labelnya?
#
# Pendekatan confident learning: baris dicurigai kalau model lintas fold yakin pada kelas lain dan tidak yakin pada
# label yang diberikan. Hasilnya hanya perkiraan kasar, jadi baris tidak dibuang otomatis. Daftarnya disimpan untuk
# diperiksa manual.

# %%
linear = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
noise_probs = np.zeros((len(data.y), n_classes))
for train_index, val_index in splitter.split(labeled_vectors, data.y):
    model = linear.fit(labeled_vectors[train_index], data.y[train_index])
    noise_probs[np.ix_(val_index, model.classes_)] = model.predict_proba(labeled_vectors[val_index])
given = noise_probs[np.arange(len(data.y)), data.y]
class_mean = np.array([noise_probs[data.y == c, c].mean() if (data.y == c).any() else 1 for c in range(n_classes)])
masked = noise_probs.copy()
masked[np.arange(len(data.y)), data.y] = 0
suspect = (given < class_mean[data.y]) & (masked.max(1) >= class_mean[masked.argmax(1)]) & (noise_probs.argmax(1) != data.y)
suspects = data.labeled.loc[suspect, ["text", "label"]].assign(
    disangka=[data.classes[i] for i in noise_probs[suspect].argmax(1)], peluang_label=given[suspect].round(3)
).sort_values("peluang_label")
suspects.to_csv("hasil/label_meragukan.csv", index=False)
display(suspects.head(5))
severity = 2 if suspect.mean() >= 0.2 else 0
notes.add(f"Perkiraan kasar {suspect.mean():.1%} label meragukan", "Baris tidak dibuang. Label smoothing diuji di 8.1, dan daftarnya ada di hasil/label_meragukan.csv", severity)

# %% [markdown]
# ### 4.13 Tambah data masih menaikkan skor?
#
# Kurva belajar model linear. Kalau skor masih naik saat label ditambah, data tambahan dari pseudo-label berpeluang
# membantu. Kalau sudah datar, masalahnya bukan jumlah data.

# %%
fractions = [0.25, 0.5, 0.75, 1.0]
curve = {fraction: [] for fraction in fractions}
for fit_index, val_index in ShuffleSplit(3, test_size=0.25, random_state=SEED).split(labeled_vectors):
    for fraction in fractions:
        subset = fit_index[: max(n_classes * 2, int(fraction * len(fit_index)))]
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(labeled_vectors[subset], data.y[subset])
        curve[fraction].append(score(data.y[val_index], model.predict(labeled_vectors[val_index])))
curve_mean = pd.Series({fraction: np.mean(values) for fraction, values in curve.items()})
curve_mean.plot(marker="o", title="Kurva belajar model linear", xlabel="porsi data latih", ylabel=METRIC)
plt.tight_layout()
plt.show()
slope = curve_mean[1.0] - curve_mean[0.5]
print(curve_mean.round(4).to_dict())
if slope >= 0.01:
    notes.add(f"Skor masih naik {slope:+.3f} dari setengah ke seluruh label", "Data tambahan berpeluang membantu, self-training layak dicoba")
else:
    notes.add(f"Kurva belajar datar, {slope:+.3f}", "Tambahan data dari pseudo-label kemungkinan kecil dampaknya", 1)

# %% [markdown]
# ### 4.14 Temuan dan keputusan
#
# Semua temuan di atas diurutkan menurut bobotnya. Baris paling atas adalah masalah terbesar pada data ini.

# %%
print("MASALAH TERBESAR:", notes.biggest)
notes.table()

# %% [markdown]
# ## 5. Preprocessing
#
# Preprocessing dibagi dua. Langkah wajib membuang gangguan teknis. Langkah opsional dipicu oleh temuan EDA, lalu harus
# lolos uji ablasi. TF-IDF dan transformer punya kebutuhan berbeda, jadi masing-masing punya versi teks sendiri:
# `clean` untuk transformer dan `sparse` untuk TF-IDF.
#
# ### 5.1 Langkah wajib **(wajib)**
#
# Mendekode emoji byte, membuang HTML, URL, mention, dan token anonim, lalu menjadikan huruf kecil. Baris latih yang
# kosong setelah ini dibuang. Baris test tetap ada.

# %%
changed = (data.labeled["text"] != data.labeled["clean"]).mean()
different = data.labeled[data.labeled["text"].str.lower().str.strip() != data.labeled["clean"]]
if len(different):
    display(different[["text", "clean"]].head(3))
print(f"baris berubah: {changed:.1%}, berubah selain huruf besar: {len(different) / len(data.labeled):.1%}")
before = len(data.labeled), len(data.unlabeled)
data.labeled = data.labeled[data.labeled["clean"].str.len() > 0].reset_index(drop=True)
data.unlabeled = data.unlabeled[data.unlabeled["clean"].str.len() > 0].reset_index(drop=True)
print(f"dibuang karena kosong: {before[0] - len(data.labeled)} berlabel, {before[1] - len(data.unlabeled)} tanpa label")

# %% [markdown]
# ### 5.2 Teks sama dengan label berbeda
#
# Label diganti menjadi suara terbanyak. Kalau seri, barisnya dibuang karena tidak ada informasi yang bisa dipercaya.

# %%
conflicted =data.labeled.groupby("clean")["label"].transform("nunique") > 1
if conflicted.any():
    top_label = data.labeled[conflicted].groupby("clean")["label"].agg(lambda s: s.mode()[0] if len(s.mode()) == 1 else None)
    resolved = data.labeled["clean"].map(top_label)
    use = conflicted & resolved.notna()
    data.labeled.loc[use, "label"] = resolved[use]
    data.labeled = data.labeled[~(conflicted & resolved.isna())].reset_index(drop=True)
print(f"teks berkonflik: {int(conflicted.sum())} baris | sisa berlabel: {len(data.labeled)}")

# %% [markdown]
# **Wajib.** Grup near-duplicate dihitung setelah label bersih, lalu dibuat 3 fold untuk uji cepat di 5.3, 5.4, dan 8.1.

# %%
steps = nk.make_steps(nk.load_slang(MODEL_DIR))
if "group" not in data.labeled.columns:
    data.labeled["group"] = nk.near_duplicate_groups(data.labeled["clean"].tolist())
print(f"grup validasi: {data.labeled['group'].nunique()} dari {len(data.labeled)} baris")
quick_folds = nk.make_folds(data, 3, SEED)
y = data.y

# %% [markdown]
# ### 5.3 Ablasi langkah opsional untuk TF-IDF
#
# Langkah dicoba satu per satu. Langkah dipakai hanya kalau peluang lebih baiknya minimal 0,9 dibanding kondisi saat
# itu. Uji ini memakai data berlabel yang sama dengan yang dilaporkan, jadi skor akhir sedikit optimistis.

# %%
def sparse_oof(names):
    texts = nk.apply_steps(data.labeled["clean"], steps, names)
    learner = nk.SparseLearner("lr", texts, n_classes)
    return nk.cross_fit("ablasi", learner.fit_predict, texts, y, quick_folds, n_classes=n_classes, check_time=False).oof.argmax(1)


begin = time.time()
current = sparse_oof([])
rows = [{"langkah": "tanpa langkah opsional", "skor": round(score(y, current), 4), "selisih": 0.0, "peluang": None, "dipakai": True}]
for step in settings.candidate_steps:
    if time.time() - begin > 0.1 * clock.budget:
        print(f"batas waktu ablasi tercapai, {step} dan seterusnya dilewati")
        break
    trial = sparse_oof([*settings.sparse_steps, step])
    chance = nk.prob_better(y, current, trial, score)
    accepted = chance >= 0.9
    rows.append({"langkah": step, "skor": round(score(y, trial), 4), "selisih": round(score(y, trial) - score(y, current), 4), "peluang": chance, "dipakai": accepted})
    if accepted:
        settings.sparse_steps.append(step)
        current = trial
display(pd.DataFrame(rows))
print("langkah untuk TF-IDF:", settings.sparse_steps or "tidak ada")

# %% [markdown]
# ### 5.4 Ablasi normalisasi untuk transformer
#
# Hasil TF-IDF belum tentu berlaku untuk transformer, jadi normalisasi diuji langsung pada model utama dengan 3 fold dan
# epoch lebih sedikit. Tanda baca dan stopword tidak diuji, karena transformer butuh kalimat utuh. Sel ini dilewati kalau
# waktu sempit.

# %%
normal_steps = [s for s in settings.candidate_steps if s in {"karakter_ulang", "tawa", "hashtag", "slang"}]
if usable_models and normal_steps and WAKTU_MENIT >= 15:
    quick = nk.Trainer(MODEL_DIR, n_classes, settings, [])
    epochs = min(plan.epochs, 4)

    def transformer_oof(names):
        texts = nk.apply_steps(data.labeled["clean"], steps, names)
        fit = quick.learner(usable_models[0], epochs, SEED)
        return nk.cross_fit("ablasi", fit, texts, y, quick_folds, n_classes=n_classes, check_time=False).oof.argmax(1)

    plain, normalized = transformer_oof([]), transformer_oof(normal_steps)
    chance = nk.prob_better(y, plain, normalized, score)
    print(f"tanpa normalisasi {score(y, plain):.4f} | dengan {normal_steps} {score(y, normalized):.4f} | peluang lebih baik {chance:.2f}")
    if chance >= 0.9:
        settings.transformer_steps = normal_steps
    notes.add(f"Normalisasi pada transformer: peluang lebih baik {chance:.2f}", "Dipakai" if chance >= 0.9 else "Tidak dipakai, teks transformer tetap apa adanya")
else:
    print("uji transformer dilewati")

# %% [markdown]
# ### 5.5 Terapkan **(wajib)**
#
# Terapkan hasil ablasi ke semua bagian data, lalu tampilkan contoh hasil akhirnya.

# %%
for frame in (data.labeled, data.unlabeled, data.test):
    frame["sparse"] = nk.apply_steps(frame["clean"], steps, settings.sparse_steps)
    frame["clean"] = nk.apply_steps(frame["clean"], steps, settings.transformer_steps)
print("transformer:", settings.transformer_steps or "hanya langkah wajib", "| TF-IDF:", settings.sparse_steps or "hanya langkah wajib")
print(f"max_len {settings.train.max_len} | potong awal dan akhir: {settings.train.head_tail}")
data.labeled[["text", "clean", "sparse"]].head(3)

# %% [markdown]
# ## 6. Validasi
#
# ### 6.1 Metrik diuji dulu **(wajib)**
#
# Metrik diuji pada contoh yang jawabannya sudah dihitung manual.

# %%
nk.check_metrics()

# %% [markdown]
# ### 6.2 Fold dan papan skor **(wajib)**
#
# `StratifiedGroupKFold`: proporsi kelas terjaga, dan grup near-duplicate tidak lintas fold. Tiap fold melatih teacher dan
# student hanya dari fold latih, jadi label validasi tidak bocor lewat pseudo-label. Label test tidak pernah dipakai
# memilih apa pun.

# %%
folds = nk.make_folds(data, plan.folds, SEED)
y = data.y
board = nk.Leaderboard(data, METRIC, SEED)
display(pd.crosstab(pd.Series(folds, name="fold"), data.labeled["label"].to_numpy(), colnames=["kelas"]))
effective = board.weights.sum() ** 2 / (board.weights**2).sum()
print(f"ukuran efektif skor mirip test: {effective:.0f} dari {len(y)} baris")
if effective < 0.3 * len(y):
    notes.add(f"Ukuran efektif skor mirip test hanya {effective:.0f} baris", "Selisih skor mirip test sangat berisik", 1)

# %% [markdown]
# ### 6.3 Batas noise
#
# Model yang sama dilatih dengan pembagian fold berbeda. Selisih skor akibat pembagian saja menjadi patokan: perbedaan di
# bawah angka ini tidak berarti apa-apa.

# %%
noise_scores = []
for seed in range(SEED, SEED + 3):
    other_folds = nk.make_folds(data, 3, seed)
    learner = nk.SparseLearner("lr", data.labeled["sparse"].tolist(), n_classes)
    preds = nk.cross_fit("noise", learner.fit_predict, data.labeled["sparse"], y, other_folds, n_classes=n_classes, check_time=False)
    noise_scores.append(score(y, preds.oof.argmax(1)))
settings.noise_floor = float(np.std(noise_scores))
print(f"skor per pembagian: {np.round(noise_scores, 4).tolist()} | batas noise (sd): {settings.noise_floor:.4f}")

# %% [markdown]
# ## 7. Baseline
#
# ### 7.1 Tebakan mayoritas dan model linear **(wajib)**
#
# Regresi logistik, SVM linear, dan Complement Naive Bayes pada TF-IDF kata dan karakter. Ketiganya tidak butuh file dari
# luar, jadi menjadi cadangan kalau pretrained dilarang atau waktu habis.

# %%
majority = np.bincount(y).argmax()
board.record("mayoritas", np.full(len(y), majority), np.full(len(data.test), majority))
sparse_texts = data.labeled["sparse"].tolist()
sparse_test = data.test["sparse"].tolist()
vocabulary_texts = sparse_texts + data.unlabeled["sparse"].tolist() + (sparse_test if PAKAI_TEST_SAAT_LATIH else [])
baselines = {}
with nk.stage("baseline_linear"):
    for kind, label in [("lr", "tfidf_lr"), ("svm", "tfidf_svm"), ("nb", "tfidf_nb")]:
        learner = nk.SparseLearner(kind, vocabulary_texts, n_classes)
        preds = nk.cross_fit(label, learner.fit_predict, sparse_texts, y, folds, sparse_test, n_classes=n_classes, check_time=False)
        baselines[label] = board.add(preds)

# %% [markdown]
# ### 7.2 Embedding + regresi logistik
#
# Model pretrained tidak dilatih ulang, hanya dipakai sebagai pengubah teks menjadi vektor. Pada label sedikit, cara
# cepat ini sering sulit dikalahkan, jadi menjadi lawan yang adil untuk fine-tuning.

# %%
if usable_models:
    all_test_vectors = np.asarray(represent(data.test["clean"].tolist()))
    labeled_vectors = np.asarray(represent(data.labeled["clean"].tolist()))
    oof, test = np.zeros((len(y), n_classes), np.float32), np.zeros((len(data.test), n_classes), np.float32)
    for fold in np.unique(folds):
        fit_mask = folds != fold
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(labeled_vectors[fit_mask], y[fit_mask])
        oof[np.ix_(~fit_mask, model.classes_)] = model.predict_proba(labeled_vectors[~fit_mask])
        test[:, model.classes_] += model.predict_proba(all_test_vectors) / plan.folds
    baselines["embedding_lr"] = board.add(nk.Preds("embedding_lr", oof, test))

# %% [markdown]
# ### 7.3 Label spreading
#
# Baseline semi-supervised tanpa pseudo-label: label menyebar ke tetangga terdekat pada graf vektor teks, termasuk lewat
# data tanpa label. Ini menjawab apakah struktur data tanpa label saja sudah membantu.

# %%
if len(data.unlabeled) and usable_models:
    labeled_vectors = np.asarray(represent(data.labeled["clean"].tolist()))
    all_test_vectors = np.asarray(represent(data.test["clean"].tolist()))
    unlabeled_vectors = np.asarray(represent(data.unlabeled["clean"].tolist()))
    oof, test = np.zeros((len(y), n_classes), np.float32), np.zeros((len(data.test), n_classes), np.float32)
    for fold in np.unique(folds):
        hidden = y.copy()
        hidden[folds == fold] = -1
        stacked = np.vstack([labeled_vectors, unlabeled_vectors, all_test_vectors])
        targets = np.concatenate([hidden, np.full(len(unlabeled_vectors) + len(all_test_vectors), -1)])
        model = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2).fit(stacked / np.linalg.norm(stacked, axis=1, keepdims=True), targets)
        spread = np.nan_to_num(model.label_distributions_)
        spread = np.where(spread.sum(1, keepdims=True) > 0, spread, 1 / n_classes)
        full = np.zeros((len(spread), n_classes), np.float32)
        full[:, model.classes_[model.classes_ >= 0]] = spread
        oof[folds == fold] = full[: len(y)][folds == fold]
        test += full[-len(all_test_vectors) :] / plan.folds
    baselines["label_spreading"] = board.add(nk.Preds("label_spreading", oof, test))

# %% [markdown]
# **Baseline terbaik** dipilih dari skor mirip test, lalu menjadi lawan pertama model utama.

# %%
best_baseline = max(baselines.values(), key=lambda p: score(y, p.oof.argmax(1), board.weights))
candidates = {best_baseline.name: best_baseline}
print("baseline terbaik:", best_baseline.name)

# %% [markdown]
# ## 8. Model Utama
#
# Self-training dalam dua lapis: teacher dilatih dari data berlabel, lalu student dilatih dari data berlabel ditambah
# pseudo-label. Pseudo-label dipilih per fold, jadi baris validasi tidak pernah memengaruhi model yang menilainya.
#
# ### 8.1 Uji setelan pelatihan
#
# Bobot kelas, label smoothing, dan layer decay diuji satu per satu pada 3 fold. Setelan diganti hanya kalau peluang lebih
# baiknya minimal 0,9. Sel ini dilewati kalau waktu sempit.

# %%
trainer = nk.Trainer(MODEL_DIR, n_classes, settings, data.labeled["clean"].tolist() + data.unlabeled["clean"].tolist())
if usable_models and WAKTU_MENIT >= 15:
    begin = time.time()
    epochs = min(plan.epochs, 4)
    fit = trainer.learner(usable_models[0], epochs, SEED)
    current = nk.cross_fit("uji", fit, data.labeled["clean"], y, quick_folds, n_classes=n_classes, check_time=False).oof.argmax(1)
    rows = [{"setelan": "awal", "skor": round(score(y, current), 4), "peluang": None, "dipakai": True}]
    for field_name, value in [("class_weight", not settings.train.class_weight), ("smoothing", 0.1), ("layer_decay", 0.9)]:
        if time.time() - begin > 0.1 * clock.budget:
            print("batas waktu uji setelan tercapai")
            break
        fit = trainer.learner(usable_models[0], epochs, SEED, **{field_name: value})
        trial = nk.cross_fit("uji", fit, data.labeled["clean"], y, quick_folds, n_classes=n_classes, check_time=False).oof.argmax(1)
        chance = nk.prob_better(y, current, trial, score)
        rows.append({"setelan": f"{field_name}={value}", "skor": round(score(y, trial), 4), "peluang": chance, "dipakai": chance >= 0.9})
        if chance >= 0.9:
            setattr(settings.train, field_name, value)
            current = trial
    display(pd.DataFrame(rows))
print(settings.train)

# %% [markdown]
# ### 8.2 Teacher **(wajib)**
#
# Tiap model dilatih pada semua fold, lalu probabilitasnya dirata-rata. Model yang tidak muat dalam sisa waktu dilewati.
# Kalau tidak ada model pretrained, langkah yang sama berjalan dengan TF-IDF.

# %%
model_names = usable_models or ["tfidf"]
texts = data.labeled["clean"].tolist()
test_texts = data.test["clean"].tolist()
pool = data.unlabeled["clean"].tolist() + (test_texts if PAKAI_TEST_SAAT_LATIH else [])
teachers = {}
with nk.stage("teacher"):
    for name in model_names:
        fit = trainer.learner(name, plan.epochs, SEED)
        preds = nk.cross_fit(f"teacher_{name}", fit, texts, y, folds, test_texts, pool, track=not teachers, n_classes=n_classes)
        if preds is None:
            break
        teachers[name] = board.add(preds)
teacher = None
if teachers:
    teacher = nk.blend("teacher", list(teachers.values()))
    if len(teachers) > 1:
        board.add(teacher)
    candidates["teacher"] = teacher

# %% [markdown]
# **Kurva validasi teacher pertama, fold pertama.** Kalau skor sudah turun jauh sebelum epoch terakhir, kurangi epoch.

# %%
first_curve = next(iter(teachers.values())).extra["curve"] if teachers else None
if first_curve:
    plt.plot(range(1, len(first_curve) + 1), first_curve, marker="o")
    plt.title("Skor validasi per epoch")
    plt.xlabel("epoch")
    plt.ylabel("macro-F1 validasi")
    plt.tight_layout()
    plt.show()
    print(f"terbaik di epoch {int(np.argmax(first_curve)) + 1}, epoch terakhir {first_curve[-1]:.4f}, puncak {max(first_curve):.4f}")
else:
    print("tidak ada kurva, model linear tidak berjalan per epoch")

# %% [markdown]
# ### 8.3 Student dari pseudo-label
#
# Aturan memilih pseudo-label untuk fold `f`:
#
# 1. **Ambang per kelas.** Dipilih dari prediksi OOF fold lain, supaya presisinya minimal `settings.target_precision`.
# 2. **Kesepakatan.** Pada ronde pertama, semua teacher harus menebak kelas yang sama.
# 3. **Kuota per kelas.** Sebanding proporsi kelas, supaya kelas besar tidak mendominasi.
# 4. **Bobot.** Baris pseudo-label diberi bobot sebesar keyakinannya.
#
# Ronde kedua memakai student ronde pertama sebagai sumber. Ronde dilewati kalau waktunya tidak cukup.

# %%
pseudo_rows, last_extra, student = [], None, None
source = teacher
with nk.stage("student"):
    for round_number in range(1, plan.rounds + 1):
        if source is None or not pool:
            print("student dilewati:", "tidak ada data tanpa label" if not pool else "teacher tidak selesai")
            break
        extra = {}
        for fold in np.unique(folds):
            other = folds != fold
            thresholds = nk.class_thresholds(source.oof[other], y[other], settings.target_precision)
            priors = np.bincount(y[other], minlength=n_classes) / other.sum()
            agree = None
            if round_number == 1 and len(teachers) > 1:
                votes = [t.extra["pool"][fold].argmax(1) for t in teachers.values()]
                agree = np.all([vote == votes[0] for vote in votes], axis=0)
            index, pseudo, confidence = nk.select_pseudo(source.extra["pool"][fold], thresholds, priors, agree)
            extra[fold] = ([pool[i] for i in index], pseudo, confidence)
            pseudo_rows.append(pd.DataFrame({"ronde": round_number, "fold": fold, "baris": index, "kelas": pseudo,
                                             "yakin": confidence, "ambang": thresholds[pseudo]}))
        print(f"ronde {round_number}: pseudo-label per fold {[len(extra[f][1]) for f in extra]}")
        members = []
        for name in teachers or model_names:
            fit = trainer.learner(name, plan.student_epochs, SEED)
            preds = nk.cross_fit(f"student_{name}_r{round_number}", fit, texts, y, folds, test_texts, pool, extra, n_classes=n_classes)
            if preds is None:
                break
            members.append(preds)
        if not members:
            break
        student = board.add(nk.blend(f"student_r{round_number}", members))
        candidates[student.name] = student
        source, last_extra = student, extra

# %% [markdown]
# ### 8.4 Seed tambahan
#
# Melatih ulang dengan seed lain mengurangi varians tanpa asumsi apa pun. Dipakai kalau sisa waktu cukup, pada tahap
# terakhir yang selesai.

# %%
with nk.stage("seed_tambahan"):
    final_member = student or teacher
    bag = []
    for name in [n for n in teachers if n != "tfidf"] if final_member is not None else []:
        epochs = plan.student_epochs if student else plan.epochs
        fit = trainer.learner(name, epochs, SEED + 1)
        preds = nk.cross_fit(f"seed2_{name}", fit, texts, y, folds, test_texts, pool, last_extra, n_classes=n_classes)
        if preds is None:
            print("sisa waktu tidak cukup untuk seed tambahan")
            break
        bag.append(preds)
    if bag:
        bagged = nk.blend(f"{final_member.name}_2seed", [final_member, nk.blend("seed2", bag)])
        candidates[bagged.name] = board.add(bagged)

# %% [markdown]
# ### 8.5 Gabungan
#
# Rata-rata sama rata antara model dengan cara belajar berbeda. Tidak ada bobot yang difit, jadi tidak bisa overfit ke OOF.

# %%
if teacher is not None and student is not None:
    candidates["gabung_teacher_student"] = board.add(nk.blend("gabung_teacher_student", [teacher, student]))
if teacher is not None and best_baseline.name != teacher.name:
    deep = student or teacher
    candidates["gabung_dengan_baseline"] = board.add(nk.Preds("gabung_dengan_baseline", (deep.oof + best_baseline.oof) / 2, (deep.test + best_baseline.test) / 2))

# %% [markdown]
# ## 9. Post-processing
#
# ### 9.1 Gerbang pemilihan model **(wajib)**
#
# Mulai dari baseline. Kandidat berikutnya menggantikan model saat ini hanya kalau peluang lebih baiknya minimal 0,9 pada
# skor mirip test. Tabel juga menunjukkan berapa fold yang dimenangkan, karena kemenangan dari satu fold saja tidak stabil.

# %%
names = list(candidates)
best = candidates[names[0]]
rows = []
for name in names[1:]:
    rival = candidates[name]
    chance = board.prob_better(best.oof.argmax(1), rival.oof.argmax(1))
    wins = sum(score(y[folds == f], rival.oof[folds == f].argmax(1)) > score(y[folds == f], best.oof[folds == f].argmax(1)) for f in np.unique(folds))
    rows.append({"kandidat": name, "lawan": best.name, "peluang": chance, "menang_fold": f"{wins}/{len(np.unique(folds))}", "dipakai": chance >= 0.9})
    if chance >= 0.9:
        best = rival
display(pd.DataFrame(rows) if rows else "hanya ada satu kandidat")
oof_pred, test_pred, final_name = best.oof.argmax(1), best.test.argmax(1), best.name
print("model terpilih:", best.name)

# %% [markdown]
# ### 9.2 Kalibrasi
#
# Temperature scaling mengukur apakah model terlalu yakin. Suhu tidak mengubah kelas prediksi, tetapi membuat keyakinan
# bisa dibaca sebagai peluang. Suhu difit dari fold lain.

# %%
calibrated = np.zeros_like(best.oof)
for fold in np.unique(folds):
    calibrated[folds == fold] = nk.apply_temperature(best.oof[folds == fold], nk.fit_temperature(best.oof[folds != fold], y[folds != fold]))
temperature = nk.fit_temperature(best.oof, y)
reliability, ece_before = nk.calibration_table(best.oof, y)
_, ece_after = nk.calibration_table(calibrated, y)
display(reliability)
print(f"suhu {temperature:.2f} | ECE sebelum {ece_before:.3f}, sesudah {ece_after:.3f}")
best_probs_test = nk.apply_temperature(best.test, temperature)

# %% [markdown]
# ### 9.3 Pengali per kelas
#
# Probabilitas dikali satu angka per kelas supaya macro-F1 maksimal. Skornya nested: tiap fold memakai pengali dari fold
# lain. Dipakai hanya kalau test tidak bergeser, naik minimal 0,002, dan peluang lebih baiknya minimal 0,9.

# %%
if settings.has_shift:
    print(f"AUC adversarial {settings.shift_auc:.3f}: test bergeser, pengali tidak dipakai")
else:
    nested = nk.nested_multiplier_pred(best.oof, y, folds, score)
    multipliers = nk.fit_multipliers(best.oof, y, score)
    tuned_test = (best.test * multipliers).argmax(1)
    gain = score(y, nested) - score(y, oof_pred)
    chance = nk.prob_better(y, oof_pred, nested, score)
    board.record(f"{best.name}_pengali", nested, tuned_test)
    print(f"pengali {dict(zip(data.classes, multipliers.round(1)))} | kenaikan nested {gain:+.4f} | peluang {chance:.2f}")
    if gain >= 0.002 and chance >= 0.9:
        oof_pred, test_pred, final_name = nested, tuned_test, f"{best.name}_pengali"
        print("pengali dipakai")
    else:
        print("pengali tidak dipakai")

# %% [markdown]
# ### 9.4 Proporsi kelas test
#
# Metode Saerens memperkirakan proporsi kelas test tanpa label. Probabilitas bisa dikoreksi sesuai proporsi itu, tetapi
# metode ini mengasumsikan hanya proporsi yang berubah. Jadi saat teks test ikut bergeser, hasilnya hanya untuk dibaca.

# %%
train_prior = np.bincount(y, minlength=n_classes) / len(y)
source_prior = best.oof.mean(0)  # model berbobot kelas tidak mengikuti proporsi latih, jadi rujukannya rata-rata OOF
test_prior = nk.em_prior(best.test, source_prior)
display(pd.DataFrame({"latih": train_prior, "rata_rata_oof": source_prior, "perkiraan_test": test_prior}, index=data.classes).round(3))
distance = 0.5 * np.abs(test_prior - source_prior).sum()
print(f"selisih proporsi {distance:.3f}")
if not settings.has_shift and distance >= 0.05 and final_name == best.name:
    ratio_prior = test_prior / np.clip(source_prior, 1e-6, None)
    corrected_oof = (best.oof * ratio_prior).argmax(1)
    row_weights = (test_prior / np.clip(train_prior, 1e-6, None))[y]
    chance = nk.prob_better(y, oof_pred, corrected_oof, score, row_weights)
    print(f"peluang lebih baik pada proporsi test {chance:.2f}")
    if chance >= 0.9:
        board.record(f"{best.name}_koreksi_proporsi", corrected_oof, (best.test * ratio_prior).argmax(1))
        oof_pred, test_pred, final_name = corrected_oof, (best.test * ratio_prior).argmax(1), f"{best.name}_koreksi_proporsi"

# %% [markdown]
# ### 9.5 Teks kembar di test
#
# Teks test yang persis ada di data berlabel disalin labelnya, tetapi hanya kalau kumpulan teks kembar di data berlabel
# memang konsisten. Teks kembar di dalam test diberi prediksi yang sama.

# %%
test_pred = test_pred.copy()
consensus = pd.Series(best_probs_test.tolist(), index=data.test["clean"]).groupby(level=0).apply(lambda s: int(np.mean(s.tolist(), axis=0).argmax()))
repeated = data.test["clean"].duplicated(keep=False).to_numpy()
changed_internal = int((test_pred[repeated] != data.test["clean"][repeated].map(consensus).to_numpy()).sum())
if final_name == best.name:
    test_pred[repeated] = data.test["clean"][repeated].map(consensus).to_numpy()
print(f"teks kembar di test: {int(repeated.sum())} baris, {changed_internal if final_name == best.name else 0} prediksi diselaraskan")
if settings.copy_duplicates:
    purity = data.labeled.groupby("clean")["label"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    multiple = data.labeled.groupby("clean")["label"].size() > 1
    known_label = data.labeled.groupby("clean")["label"].first().map(data.class_id)
    hits = data.test["clean"].map(known_label)
    consistent = purity[multiple].mean() if multiple.any() else 1.0
    print(f"konsistensi teks kembar berlabel: {consistent:.3f}")
    if consistent >= 0.9:
        changed_rows = int((hits.notna().to_numpy() & (hits.fillna(-1).to_numpy() != test_pred)).sum())
        test_pred[hits.notna().to_numpy()] = hits.dropna().astype(int).to_numpy()
        print(f"disalin dari data berlabel: {int(hits.notna().sum())} baris, {changed_rows} prediksi berubah")

# %% [markdown]
# ### 9.6 Keputusan akhir **(wajib)**

# %%
final = nk.Final(final_name, oof_pred, test_pred, best.oof, best.test)
board.record(f"FINAL_{final.name}", final.oof_pred, final.test_pred)
print("FINAL:", final.name)

# %% [markdown]
# ## 10. Post-analysis
#
# ### 10.1 Tabel perbandingan semua pendekatan
#
# `oof` dan rentangnya dari bootstrap, `oof_mirip_test` berbobot, dan `test` hanya muncul kalau label test ada.

# %%
print(f"batas noise {settings.noise_floor:.4f}. Selisih di bawah angka ini tidak berarti.")
board.table()

# %% [markdown]
# ### 10.2 Skor per kelas dan per fold
#
# Kelas mana yang menyeret macro-F1, dan apakah skornya stabil antar fold.

# %%
per_class = []
rng = np.random.default_rng(SEED)
for index, name in enumerate(data.classes):
    tp = ((final.oof_pred == index) & (y == index)).sum()
    precision = tp / max((final.oof_pred == index).sum(), 1)
    recall = tp / max((y == index).sum(), 1)
    per_class.append({"kelas": name, "baris": int((y == index).sum()), "presisi": precision, "recall": recall,
                      "f1": 2 * precision * recall / max(precision + recall, 1e-9), "prediksi": int((final.oof_pred == index).sum())})
display(pd.DataFrame(per_class).round(3))
print("skor per fold:", {int(f): round(score(y[folds == f], final.oof_pred[folds == f]), 4) for f in np.unique(folds)})
nk.plot_confusion(data.classes, y, final.oof_pred)

# %% [markdown]
# ### 10.3 Proporsi kelas
#
# Kalau proporsi prediksi test jauh berbeda dari data latih, skor OOF kurang bisa dipercaya. Kolom asli hanya untuk audit.

# %%
share_table = {"latih": np.bincount(y, minlength=n_classes) / len(y),
               "prediksi_oof": np.bincount(final.oof_pred, minlength=n_classes) / len(y),
               "prediksi_test": np.bincount(final.test_pred, minlength=n_classes) / len(final.test_pred)}
if data.has_test_labels:
    known_test = data.y_test[data.y_test >= 0]
    share_table["asli_test"] = np.bincount(known_test, minlength=n_classes) / len(known_test)
pd.DataFrame(share_table, index=data.classes).round(3)

# %% [markdown]
# ### 10.4 Skor per kelompok data
#
# Melihat di mana model lemah: teks pendek atau panjang, mengandung emoji atau slang, dan yang paling mirip test. Kelompok
# dengan skor jauh di bawah rata-rata adalah arah perbaikan berikutnya.

# %%
frame = data.labeled
slang = nk.load_slang(MODEL_DIR)
lengths = frame["clean"].str.split().str.len()
groups = {
    "pendek": lengths <= lengths.quantile(0.33),
    "panjang": lengths >= lengths.quantile(0.67),
    "ada_emoji": frame["text"].str.contains(nk.text.EMOJI_RE) | frame["text"].str.contains(nk.text.BYTES_RE),
    "ada_slang": frame["clean"].map(lambda t: any(word in slang for word in t.split())),
    "mirip_test": pd.Series(board.weights >= np.quantile(board.weights, 0.75), index=frame.index),
    "semua": pd.Series(True, index=frame.index),
}
slices = [{"kelompok": name, "baris": int(mask.sum()), "skor": round(score(y[mask], final.oof_pred[mask]), 4)} for name, mask in groups.items() if mask.sum() >= 20]
pd.DataFrame(slices).sort_values("skor")

# %% [markdown]
# ### 10.5 Kualitas pseudo-label
#
# Jumlah, keyakinan, dan ambang per kelas. Presisi hanya muncul kalau ada kolom `label_audit`, dan kolom itu tidak pernah
# dipakai untuk memilih. Tanpa kolom itu, sampel acak di bawahnya bisa diperiksa manual.

# %%
if pseudo_rows:
    log = pd.concat(pseudo_rows, ignore_index=True)
    log["nama_kelas"] = [data.classes[i] for i in log["kelas"]]
    summary = {"jumlah": ("yakin", "size"), "yakin": ("yakin", "mean"), "ambang": ("ambang", "mean")}
    if "audit" in data.unlabeled.columns:
        truth = data.unlabeled["audit"].map(data.class_id).to_numpy()
        inside = log["baris"].to_numpy() < len(truth)
        log["benar"] = np.nan
        log.loc[inside, "benar"] = (truth[log.loc[inside, "baris"].to_numpy()] == log.loc[inside, "kelas"].to_numpy()).astype(float)
        summary["presisi"] = ("benar", "mean")
    display(log.groupby(["ronde", "nama_kelas"]).agg(**summary).round(3))
    sample = log.sample(min(5, len(log)), random_state=SEED)
    display(pd.DataFrame({"teks": [pool[i] for i in sample["baris"]], "pseudo": sample["nama_kelas"].to_numpy(), "yakin": sample["yakin"].round(3).to_numpy()}))
else:
    print("tidak ada pseudo-label")

# %% [markdown]
# ### 10.6 Contoh salah prediksi
#
# Salah dengan keyakinan tinggi biasanya menunjukkan label meragukan atau sarkasme. Salah dengan keyakinan rendah
# menunjukkan kasus yang memang ambigu.

# %%
errors = data.labeled[["text", "label"]].assign(prediksi=[data.classes[i] for i in final.oof_pred], yakin=final.oof_probs.max(1).round(3))
errors = errors[errors["label"] != errors["prediksi"]].sort_values("yakin", ascending=False)
print(f"jumlah salah: {len(errors)} dari {len(y)}")
display(errors.head(6))
errors.tail(4)

# %% [markdown]
# ### 10.7 Kesepakatan antar model
#
# Baris test tempat model utama dan baseline berbeda pendapat adalah yang paling tidak pasti.

# %%
disagree = final.test_pred != best_baseline.test.argmax(1)
print(f"model final dan {best_baseline.name} berbeda pada {disagree.mean():.1%} baris test")
if disagree.any():
    display(data.test.loc[disagree, ["text"]].assign(
        final=[data.classes[i] for i in final.test_pred[disagree]],
        baseline=[data.classes[i] for i in best_baseline.test.argmax(1)[disagree]],
    ).head(5))

# %% [markdown]
# ### 10.8 Poin untuk presentasi
#
# Disusun otomatis dari hasil run ini. Periksa dulu sebelum dipakai.

# %%
table = board.table()
row = table[table["model"] == f"FINAL_{final.name}"].iloc[0]
base_row = table[table["model"] == best_baseline.name].iloc[0]
print(f"1. Masalah terbesar: {notes.biggest}.")
print(f"2. Baseline terbaik {best_baseline.name}: OOF {base_row['oof']}. Model final {final.name}: OOF {row['oof']} (rentang {row['oof_ci95']}).")
print(f"3. Batas noise {settings.noise_floor:.4f}, dan model baru harus punya peluang lebih baik minimal 0,9.")
print(f"4. Preprocessing transformer: {settings.transformer_steps or 'hanya langkah wajib'}. TF-IDF: {settings.sparse_steps or 'hanya langkah wajib'}.")
print(f"5. Pseudo-label: {sum(len(r) for r in pseudo_rows)} baris lintas fold dan ronde." if pseudo_rows else "5. Tidak ada pseudo-label.")

# %% [markdown]
# ## 11. Keluaran
#
# File `hasil/submission.csv`, `hasil/oof.csv`, `hasil/perbandingan.csv`, `hasil/label_meragukan.csv`, dan ringkasan run.

# %%
preview = nk.save_outputs(data, final)
nk.finish("semi_supervised", data, board, needs_gpu=bool(MODELS))
preview
