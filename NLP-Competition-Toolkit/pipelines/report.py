# title: Laporan wawasan tanpa label
# for: report trend
# %% stage: setup | Setup & konfigurasi
# note: Satu-satunya sel yang wajib diubah. Jalankan paling awal.
# uses: optimization/gpu_check, optimization/seed_everything
import re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
MODELS = Path(os.environ.get("COMP_ROOT", r"C:\gemastik")) / "models"
SEED = 42
seed_everything(SEED)
env_report(MODELS)

DATA, TEXT, TIME = "train.csv", "text", "time"
K_TOPIC = 8
ISSUE = "keamanan_pangan"                        # isu yang dipantau di garis waktu

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)

# %% stage: eda | EDA
# note: Profil, duplikat, dan baris tak bermakna.
# uses: eda/eda_report, noisy_data/text_outliers
report = eda_report(df, None, TEXT, time=TIME if TIME in df else None)
flags = text_flags(df[TEXT])

# %% stage: clean | Cleaning: volume asli vs copypasta
# note: Pembeda laporan: pisahkan copypasta.
# uses: noisy_data/clean_all
d = clean_dataset(df, TEXT)
print("copypasta:", round(1 - len(d) / len(df), 4))

# %% stage: prep | Preprocessing
# note: Preset topic.
# uses: preprocessing/preprocess_all
docs = preprocess_corpus(d[TEXT], "topic")

# %% stage: model | Topik & aspek
# note: NMF untuk tema, leksikon untuk aspek. Audit presisi aturan pada 30 sampel.
# uses: modeling/topic_nmf_lda, evaluation/metrics_topic, domain/mbg_lexicon
lab, words, _ = fit_nmf(docs, k=K_TOPIC, min_df=2)
print("npmi", npmi_coherence(words, docs)[0], "diversity", topic_diversity(words))
for i, w in enumerate(words):
    print(i, w[:8])
H = tag_terms(d[TEXT])
samples = audit_aspects(d[TEXT], H, 30)
print("istilah nol hit:", term_hits(d[TEXT]).query("hit == 0").istilah.tolist()[:30])
d["aspek"] = np.where(H.any(axis=1), H.columns[H.to_numpy().argmax(1)], "lainnya")
d["topik"] = lab

# %% stage: eval | Garis waktu
# note: Lonjakan dan proporsi topik per minggu.
# uses: distribution_shift/feature_shift, domain/temporal
if TIME in d:
    v = volume_timeline(d[TIME], "W")
    print(v.query("lonjakan").head().to_string())
    tl = issue_timeline(d[TIME], H[ISSUE], "W")
    cps = change_points(tl)
    print("titik perubahan:", cps)
    t = pd.to_datetime(d[TIME], errors="coerce").dt.to_period("M")
    print(pd.crosstab(t, d.topik, normalize="index").round(2).tail(6).to_string())

# %% stage: out | Grafik & slide
# note: Data tidak boleh keluar: simpan gambar dan masukkan selagi lomba berjalan.
# uses: output/plots, output/slides
figs = [plot_bars(d.aspek.value_counts(), "aspek.png", "Aspek setelah dedup")]
if TIME in d:
    figs.append(plot_issue_timeline(tl, "timeline.png", f"Porsi {ISSUE} per minggu", MBG_EVENTS, cps))
build_slides([{"title": "Peta aspek", "bullets": [f"{len(d)} baris unik dari {len(df)}"], "image": figs[0]}], "hasil.pptx")
d.to_csv("submission.csv", index=False)
