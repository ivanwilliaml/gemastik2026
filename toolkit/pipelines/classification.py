# title: Klasifikasi satu label: biner atau multikelas
# for: binary spam fakenews multiclass sentiment stance emotion intent langdetect
# %% stage: setup | Setup & konfigurasi
# note: Satu-satunya sel yang wajib diubah. Jalankan paling awal.
# uses: optimization/gpu_check, optimization/seed_everything, optimization/race_plan
import re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
MODELS = Path(os.environ.get("COMP_ROOT", r"C:\gemastik")) / "models"
SEED = 42
seed_everything(SEED)
env_report(MODELS)
RACE_START = None                                # "13:00" saat lomba: jam menghitung gate dari sini

TRAIN, TEST, SAMPLE = "train.csv", "test.csv", "sample_submission.csv"
TEXT, TARGET, TIME = "text", "label", None      # TIME: nama kolom waktu bila ada
PREPROCESS = "tfidf"                             # none, tfidf, social, topic, transformer, indobertweet
EXPERIMENTS = [("tfidf_union", "lr"), ("tfidf_union", "svc"), ("tfidf_word", "cnb")]   # (vectorizer, model): tukar sesuka hati
FOLD_KIND, N_FOLDS = "auto", 5                   # auto, stratified, strat_group, group, time, source
USE_BERT = False                                 # True bila pretrained boleh dan lolos gerbang T105
BERT_MODEL, BERT_ROWS, MAX_LEN, BATCH = MODELS / "indobertweet", 50000, 128, 32
RESUME = False                                   # True: lanjut dari checkpoint terakhir setelah crash
CHECK_LABEL = False                              # True bila pretrained boleh: lacak sumber label
MBG_INSIGHT = True                               # bahan slide: aspek x sentimen, lokasi, tren
start_clock(RACE_START)

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
train, test = read_any(TRAIN), read_any(TEST)
print("train", train.shape, "| test", test.shape)
print(train.head(3).to_string())
EXPERIMENTS, N_FOLDS = scale_plan(len(train), EXPERIMENTS, N_FOLDS)   # data besar: otomatis lebih ringan

# %% stage: eda | EDA
# note: Profil, kosong, duplikat, panjang, target, kata khas, OOV, overlap, adversarial, saran, dan porsi bahasa daerah.
# uses: eda/eda_report, domain/language_mix
report = eda_report(train, test, TEXT, TARGET, time=TIME)
_ = language_report(train[TEXT].sample(min(100000, len(train)), random_state=SEED))
clock("EDA")

# %% stage: label | Lacak sumber label
# note: Kecocokan di atas 98% dengan satu model berarti label kemungkinan besar buatan model itu. Cek aturan dulu.
# uses: domain/label_source
if CHECK_LABEL:
    print(label_fingerprint(train[TEXT], train[TARGET], MODELS, inset_dir=MODELS.parent / "data_ext/inset").to_string())
else:
    print("CHECK_LABEL=False: lewati. Nyalakan bila pretrained diizinkan.")

# %% stage: clean | Cleaning data
# note: Kosong, duplikat, label bertentangan, dan grup untuk fold. Test tidak dibuang.
# uses: noisy_data/clean_all
train = clean_dataset(train, TEXT, TARGET)      # near_dup=True untuk copypasta
test = clean_test(test, TEXT)
groups = train["_group"].to_numpy()

# %% stage: prep | Preprocessing
# note: Satu fungsi, ganti preset di Setup, atau ubah satu langkah: preprocess_corpus(t, 'tfidf', emoji='remove').
# uses: preprocessing/preprocess_all
texts = preprocess_corpus(train[TEXT], PREPROCESS)
test_texts = preprocess_corpus(test[TEXT], PREPROCESS)
for a, b in list(zip(train[TEXT], texts))[:3]:
    print(a[:80], "->", b[:80])

# %% stage: feat | Fitur
# note: Nama vectorizer dipakai di EXPERIMENTS. Cek ukuran matriks di sini.
# uses: feature_engineering/make_vectorizer
for v in dict.fromkeys(e[0] for e in EXPERIMENTS):
    print(v, make_vectorizer(v).fit_transform(texts[:5000]).shape)

# %% stage: split | Validasi
# note: Encode label dan buat fold. Grup mencegah teks sama melintasi fold. Split waktu hanya memvalidasi sebagian baris (COV).
# uses: postprocessing/label_mapping, validation/make_folds
y, classes = encode_labels(train[TARGET])
folds = make_folds(y, groups if FOLD_KIND != "time" else None, None if TIME is None else train[TIME], kind=FOLD_KIND, n=N_FOLDS, seed=SEED)
check_folds(texts, folds)
COV = np.unique(np.concatenate([b for _, b in folds]))   # baris yang punya OOF: split waktu hanya memvalidasi sebagian
print("kelas", list(classes), "| fold", [len(b) for _, b in folds], "| baris tervalidasi", len(COV), "dari", len(y))
clock("validasi")

# %% stage: model | Model
# note: Melatih semua di EXPERIMENTS dengan fold yang sama, lalu memilih yang terbaik.
# uses: modeling/make_model, evaluation/metrics_classification, optimization/save_artifacts
from sklearn.metrics import f1_score
macro = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
results = {}
print("baseline", cls_report(y[COV], oof_predict(lambda: make_vectorizer("count"), lambda: make_model("dummy"), texts, y, folds)[0][COV]))
for v, m in EXPERIMENTS:
    oof, pred = oof_predict(lambda: make_vectorizer(v), lambda: make_model(m), texts, y, folds, test_texts)
    results[f"{v}+{m}"] = (oof, pred)
    save_run(f"{v}+{m}", oof, pred, macro(y[COV], oof[COV].argmax(1)))
    print(f"{v}+{m}", cls_report(y[COV], oof[COV]))
best = max(results, key=lambda k: macro(y[COV], results[k][0][COV].argmax(1)))
oof, pred = results[best]
print("terbaik:", best)
clock("baseline")

# %% stage: bert | Transformer
# note: Lewati bila USE_BERT=False. Satu holdout, bukan 5 fold. Hasil TF-IDF tetap jadi fallback.
# uses: modeling/transformer_trainer, optimization/subsample_stratified
bert_valid, bert_test = None, None
if USE_BERT:
    bt, bt_test = preprocess_corpus(train[TEXT], "indobertweet"), preprocess_corpus(test[TEXT], "indobertweet")
    tr, va = folds[0]
    tr = tr[subsample_idx(y[tr], BERT_ROWS, SEED)]
    model, tok = train_transformer(BERT_MODEL, [bt[i] for i in tr], y[tr], [bt[i] for i in va[:8000]], y[va[:8000]], num_labels=len(classes), max_len=MAX_LEN, batch=BATCH, resume=RESUME)
    bert_valid = (va, predict_transformer(model, tok, [bt[i] for i in va], max_len=MAX_LEN))
    bert_test = predict_transformer(model, tok, bt_test, max_len=MAX_LEN)
    full = np.full((len(y), bert_valid[1].shape[1]), np.nan, dtype="float32"); full[va] = bert_valid[1]
    save_run("bert", full, bert_test, macro(y[va], bert_valid[1].argmax(1)))   # aman walau kernel mati setelah ini
    print("BERT holdout", cls_report(y[va], bert_valid[1]), "\nTF-IDF holdout", cls_report(y[va], oof[va]))
else:
    print("USE_BERT=False: lanjut dengan TF-IDF")

# %% stage: eval | Evaluasi & error analysis
# note: Noise floor menentukan batas terima. Baca kesalahan sebelum menambah model.
# uses: evaluation/noise_floor, evaluation/error_analysis
nf = COV[sample_idx(len(COV), 100000, SEED)]        # bootstrap pada sampel: data penuh terlalu lambat
print("skor, SE, batas bawah, batas atas:", noise_floor(y[nf], oof[nf].argmax(1), macro, 200, groups=None))
print(per_label(y[COV], oof[COV].argmax(1), list(classes)).to_string())
print("tertukar:", confused_pairs(y[COV], oof[COV].argmax(1), list(classes), 5))
print(worst_errors(train[TEXT].iloc[COV].tolist(), y[COV], oof[COV], 10, classes).to_string())

# %% stage: post | Post-processing
# note: Blend bila gain melebihi 2 SE, lalu threshold atau pengali dari OOF.
# uses: postprocessing/prob_average, imbalance/threshold_tuning, postprocessing/class_thresholds
if bert_valid is not None:
    va, vp = bert_valid
    g = paired_gain(y[va], oof[va].argmax(1), blend([oof[va], vp]).argmax(1), macro)
    print("gain blend", g)
    if g["gain"] > 2 * g["se"]:
        pred = blend([pred, bert_test])
if len(classes) == 2:
    f_nested, thr = nested_threshold_score(y[COV], oof[COV, 1])
    print("F1 nested", f_nested, "| threshold", thr)
    final = (pred[:, 1] >= thr).astype(int)
else:
    pp = COV[sample_idx(len(COV), 100000, SEED)]
    base, nested, w = nested_multiplier_score(y[pp], oof[pp])
    print("macro-F1 argmax", base, "| nested", nested)
    final = (pred * w).argmax(1) if nested > base + .002 else pred.argmax(1)

# %% stage: out | Submission & grafik
# note: Validasi kolom, jumlah baris, NaN, dan nilai label. Gambar untuk slide.
# uses: output/plots
sub = write_submission(SAMPLE, classes[final], allowed=classes)
print(plot_confusion(y[COV], oof[COV].argmax(1), list(classes)))
print(leaderboard().to_string())
clock("submission")

# %% stage: insight | Bahan slide MBG
# note: Heatmap aspek x sentimen, lokasi, dan tren dari label train dan prediksi test. Baca audit sebelum dipakai: kamus adalah hipotesis.
# uses: domain/mbg_lexicon, domain/wilayah, domain/temporal, output/slides
if MBG_INSIGHT:
    all_text = pd.concat([train[TEXT], test[TEXT]], ignore_index=True)
    all_sent = np.concatenate([np.asarray(classes)[y], np.asarray(classes)[final]])   # label train + prediksi test
    H = tag_terms(all_text)
    print(H.mean().round(3).to_string())
    print("istilah nol hit:", term_hits(all_text.sample(min(50000, len(all_text)), random_state=SEED)).query("hit == 0").istilah.tolist()[:40])
    tbl = aspect_sentiment(H, all_sent)
    print(tbl.to_string())
    imgs = {"Aspek x sentimen": plot_heatmap(tbl, "aspek_sentimen.png", normalize=True)} if tbl.to_numpy().sum() else {}
    L = tag_locations(all_text)
    if L.provinsi.notna().any():
        imgs["Sebutan provinsi"] = plot_province_map(L.provinsi.value_counts(), MODELS.parent / "data_ext/wilayah/idn_adm1.geojson", "peta.png", "Provinsi yang disebut")
    if TIME and TIME in train and TIME in test:
        times = pd.concat([train[TIME], test[TIME]], ignore_index=True)
        tl = issue_timeline(times, H["keamanan_pangan"], "W")
        imgs["Tren keamanan pangan"] = plot_issue_timeline(tl, "tren.png", "Porsi keamanan pangan per minggu", MBG_EVENTS, change_points(tl))
    if imgs:
        build_slides([{"title": k, "bullets": [], "image": v} for k, v in imgs.items()], "bahan_slide.pptx")
    print("gambar:", list(imgs.values()))
else:
    print("MBG_INSIGHT=False")
