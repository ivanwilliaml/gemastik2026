# title: Ekstraksi informasi dan peta isu
# for: mbg_extract
# %% stage: setup | Setup & konfigurasi
# note: Satu-satunya sel yang wajib diubah. Tidak butuh label dan tidak butuh GPU, kecuali USE_NER.
# uses: optimization/gpu_check, optimization/seed_everything, optimization/race_plan
import re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
MODELS = Path(os.environ.get("COMP_ROOT", r"C:\gemastik")) / "models"
SEED = 42
seed_everything(SEED)
env_report(MODELS)

DATA, TEXT, TIME, SENTIMENT = "train.csv", "text", "time", None   # SENTIMENT: kolom sentimen bila ada
ISSUES = ["keamanan_pangan", "anggaran", "korupsi_tata_kelola"]   # isu yang dipantau di garis waktu
FREQ = "W"                                       # D, W, atau M
USE_NER, NER_MODEL, NER_ROWS = False, MODELS / "nusabert-ner", 3000
GEOJSON = MODELS.parent / "data_ext/wilayah/idn_adm1.geojson"
start_clock(None)                                # "13:00" saat lomba

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
df = read_any(DATA)
print(df.shape)
print(df.head(3).to_string())

# %% stage: eda | EDA & bahasa
# note: Profil, duplikat, dan porsi 10 bahasa daerah.
# uses: eda/eda_report, domain/language_mix
report = eda_report(df, None, TEXT, time=TIME if TIME in df else None)
_ = language_report(df[TEXT].sample(min(100000, len(df)), random_state=SEED))
clock("EDA")

# %% stage: clean | Cleaning: volume asli vs copypasta
# note: Teks identik dihitung sekali untuk peta isu. Laporkan porsi copypasta sebagai temuan.
# uses: noisy_data/clean_all
d = clean_dataset(df, TEXT).reset_index(drop=True)
print("baris unik", len(d), "| copypasta", round(1 - len(d) / len(df), 4))

# %% stage: tag | Peta isu dengan kamus
# note: Multi-label. Baca audit dan buang istilah yang salah sasaran: kamus adalah hipotesis.
# uses: domain/mbg_lexicon
H = tag_terms(d[TEXT])
print(H.sum().sort_values(ascending=False).to_string())
hits = term_hits(d[TEXT])
print(hits.sort_values("hit", ascending=False).head(25).to_string())
print("nol hit:", hits.query("hit == 0").istilah.tolist()[:40])
audit = audit_aspects(d[TEXT], H, 30)
for k in ISSUES:
    print("--", k, *[t[:90] for t in audit.get(k, [])[:5]], sep="\n   ")

# %% stage: extract | Ekstraksi terstruktur
# note: Uang, jumlah, tanggal, makanan, gejala, insiden, entitas. Angka komentar bukan data resmi.
# uses: domain/extract_structured
rec = extract_records(d[TEXT].tolist(), d[TIME].tolist() if TIME in d else None)
print({c: int(rec[c].map(len).gt(0).sum()) for c in rec})
print(pd.Series([x for r in rec.entitas for x in r]).value_counts().head(12).to_string())
print(pd.Series([x for r in rec.makanan for x in r]).value_counts().head(12).to_string())

# %% stage: geo | Lokasi & peta
# note: Sebutan lokasi, bukan bukti lokasi kejadian. Saring insiden untuk peta kejadian.
# uses: domain/wilayah
L = tag_locations(d[TEXT])
print("baris menyebut lokasi:", round(L.provinsi.notna().mean(), 4))
print(L.kab_kota.value_counts().head(15).to_string())
incident = rec.insiden.map(len).gt(0).to_numpy()
prov_all = L.provinsi.value_counts()
prov_inc = L.provinsi[incident].value_counts()
peta = plot_province_map(prov_inc if len(prov_inc) >= 3 else prov_all, GEOJSON, "peta.png", "Provinsi pada komentar insiden" if len(prov_inc) >= 3 else "Provinsi yang disebut")

# %% stage: ner | NER transformer (opsional)
# note: Lewati bila USE_NER=False. Hanya pada baris insiden agar cepat, lalu petakan ke nama baku.
# uses:
if USE_NER:
    idx = np.where(incident)[0][:NER_ROWS]
    ents = ner_entities(d[TEXT].iloc[idx].tolist(), NER_MODEL)
    print(pd.Series([e[0] for r in ents for e in r]).value_counts().to_string())
    locs = [resolve_location(w) for r in ents for g, w in r if g in ("GPE", "LOC", "PLACE", "LOCATION", "FAC")]
    print(pd.Series([x[0] for x in locs if x[0]]).value_counts().head(10).to_string())
else:
    print("USE_NER=False: pakai gazetteer saja")

# %% stage: trend | Tren isu & titik perubahan
# note: Proporsi per periode, bukan jumlah. Titik perubahan dengan ruptures, kejadian dari MBG_EVENTS.
# uses: domain/temporal
imgs = {"Peta lokasi": peta}
if TIME in d:
    for k in ISSUES:
        tl = issue_timeline(d[TIME], H[k], FREQ)
        cps = change_points(tl)
        print(k, "titik perubahan:", [str(c) for c in cps])
        print(tl[tl.cukup].tail(6).to_string())
        imgs["Tren " + k] = plot_issue_timeline(tl, f"tren_{k}.png", f"Porsi {k}", MBG_EVENTS, cps)
    print(share_over_time(d[TIME], np.where(H.any(axis=1), H.columns[H.to_numpy().argmax(1)], "lainnya"), "M").tail(6).to_string())

# %% stage: insight | Tabel & bahan slide
# note: Satu baris per dokumen untuk analisis lanjut, plus slide. Data tidak boleh keluar: masukkan selagi lomba berjalan.
# uses: output/plots, output/slides
out = pd.concat([d[[TEXT]], L[["kab_kota", "provinsi"]], rec, H.add_prefix("isu_")], axis=1)
out.astype({c: str for c in rec.columns}).to_parquet("hasil_ekstraksi.parquet", index=False)
if SENTIMENT and SENTIMENT in d:
    imgs["Aspek x sentimen"] = plot_heatmap(aspect_sentiment(H, d[SENTIMENT].astype(str).to_numpy()), "aspek_sentimen.png", normalize=True)
imgs["Peta isu"] = plot_bars(H.sum(), "isu.png", "Jumlah baris unik per isu")
build_slides([{"title": k, "bullets": [f"{len(d)} baris unik dari {len(df)}"], "image": v} for k, v in imgs.items()], "bahan_slide.pptx")
print(out.shape, list(imgs.values()))
clock("selesai")
