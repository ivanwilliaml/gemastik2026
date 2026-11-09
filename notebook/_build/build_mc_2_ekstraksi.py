"""Notebook 2 · multiclass dengan encoder pretrained yang dibekukan (feature extraction) + kepala linear / MLP."""
from mc_common import Book, head, tail, finalize

book = Book()
head(book, "Multiclass 2 · Pretrained sebagai ekstraktor fitur (backbone beku)", """Lima encoder pretrained **tidak dilatih ulang**. Teks masuk sekali, vektor kalimat diambil, lalu hanya kepala sederhana (Logistic Regression, Linear SVM, MLP kecil) yang belajar dari train. Murah, stabil, dan tidak perlu mengatur learning rate.

**Diagram alur, daftar metode, dan kapan memilih apa:** [alur_multi_kelas.html](alur_multi_kelas.html)

| Langkah | Isi |
|---|---|
| 1-2 | Konfigurasi, muat data, audit kualitas |
| 3 | Preprocessing universal (6 view) |
| 4 | Grup duplikat dan split train / validation / holdout |
| 5 | EDA angka |
| 6 | Ekstraksi embedding (mean dan CLS), ablasi pooling, tiga kepala, concat semua encoder |
| 7+ | Pilih di validation, post-processing, holdout sekali, analisis kelemahan |

**Data contoh:** [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program). Jalankan di Kaggle **GPU T4**, internet aktif (model diunduh dari Hugging Face).""", gpu=True)

book.md("""## 6. Ekstraksi embedding

| Encoder | Jenis | Catatan |
|---|---|---|
| `intfloat/multilingual-e5-base` | sentence embedding multilingual | dilatih kontrastif; wajib awalan `query: ` |
| `paraphrase-multilingual-mpnet-base-v2` | sentence embedding multilingual | dilatih parafrase |
| `indolem/indobertweet-base-uncased` | BERT bahasa Indonesia, domain Twitter | bukan sentence encoder, tapi dekat domain medsos |
| `indobenchmark/indobert-base-p1` | BERT bahasa Indonesia, domain umum | bukan sentence encoder |
| `xlm-roberta-base` | BERT multilingual | pembanding: biasanya lemah kalau dibekukan |

Dua cara menggabungkan token jadi satu vektor, keduanya disimpan: **mean** (rata-rata token berbobot mask) dan **CLS** (token pertama). Fp16 dipakai agar cepat; teks lebih dari `MAX_LEN` token dipotong, persentasenya dilaporkan.""")
book.config("`ENCODERS`: nama → (model, awalan, view teks). View `bert` dan `embed` dijelaskan di bagian 3. Tambah encoder dengan satu baris.", r'''
    ENCODERS = {"e5_base":      ("intfloat/multilingual-e5-base", "query: ", "embed"),
                "mpnet_multi":  ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "embed"),
                "indobertweet": ("indolem/indobertweet-base-uncased", "", "bert"),
                "indobert_p1":  ("indobenchmark/indobert-base-p1", "", "embed"),
                "xlm_roberta":  ("xlm-roberta-base", "", "embed")}
    MAX_LEN, BATCH = 128, 64
''')
book.code(r'''
    from transformers import AutoTokenizer, AutoModel
    emb, trunc = {}, {}
    for name, (mid, prefix, view) in ENCODERS.items():
        t0 = time.perf_counter(); tk, enc = AutoTokenizer.from_pretrained(mid), AutoModel.from_pretrained(mid).half().cuda().eval()
        trunc[name] = {"view": view, "token_p50": np.median(lens := [len(x) for x in tk([prefix + s for s in train[view]])["input_ids"]]),
                       "token_p95": np.percentile(lens, 95), "terpotong_%": 100 * np.mean(np.array(lens) > MAX_LEN)}
        for split, d in (("train", train), ("val", val), ("hold", holdout)):
            mean, cls = [], []
            for i in range(0, len(d), BATCH):
                b = tk([prefix + s for s in d[view].iloc[i:i + BATCH]], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
                with torch.no_grad(): h = enc(**b).last_hidden_state.float()
                msk = b["attention_mask"].unsqueeze(-1); mean.append(((h * msk).sum(1) / msk.sum(1)).cpu().numpy()); cls.append(h[:, 0].cpu().numpy())
            emb[name, "mean", split], emb[name, "cls", split] = np.vstack(mean), np.vstack(cls)
        trunc[name]["detik"] = time.perf_counter() - t0
        del enc; torch.cuda.empty_cache()
    show("Ekstraksi · panjang token dan waktu", pd.DataFrame(trunc).T)
    show("Dimensi embedding", pd.DataFrame({n: {"dim": emb[n, "mean", "train"].shape[1]} for n in ENCODERS}).T)
''')
book.md("### 6.1 Ablasi pooling dan tampilan\n\nSatu kepala (LR, `C=1`) pada tiap encoder dengan pooling mean vs CLS. Pooling yang menang berbeda antar encoder: sentence encoder umumnya mean, BERT biasa sering lebih baik dengan mean daripada CLS.")
book.code(r'''
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    pool_abl = {}
    for n in ENCODERS:
        pool_abl[n] = {}
        for pool in ["mean", "cls"]:
            sc = StandardScaler().fit(emb[n, pool, "train"])
            lr = LogisticRegression(C=1, max_iter=2000, class_weight=CLASS_WEIGHT).fit(sc.transform(emb[n, pool, "train"]), y_train)
            pool_abl[n][pool] = mac(y_val, lr.predict(sc.transform(emb[n, pool, "val"])))
    pool_abl = pd.DataFrame(pool_abl).T; pool_abl["terbaik"] = pool_abl.idxmax(axis=1)
    show("Ablasi pooling · LR C=1 · macro-F1 validation", pool_abl)
''')
book.md("### 6.2 Fitur untuk kepala\n\nSetiap encoder memakai pooling terbaiknya. `concat_semua` menempelkan semua encoder. Scaler dipelajari dari train saja.")
book.code(r'''
    X = {}
    for n in ENCODERS:
        sc = StandardScaler().fit(emb[n, pool_abl.terbaik[n], "train"]); X[n] = [sc.transform(emb[n, pool_abl.terbaik[n], s]) for s in ("train", "val", "hold")]
    cat = [np.hstack([emb[n, pool_abl.terbaik[n], s] for n in ENCODERS]) for s in ("train", "val", "hold")]
    sc = StandardScaler().fit(cat[0]); X["concat_semua"] = [sc.transform(x) for x in cat]
    show("Bentuk matriks", pd.DataFrame({n: {"train": x[0].shape, "val": x[1].shape, "holdout": x[2].shape} for n, x in X.items()}).T)
''')
book.md("""### 6.3 Sapuan kepala dan parameter

Tiga kepala di atas fitur beku. `C` kecil = regularisasi kuat; untuk MLP, `alpha = 1 / (100 · C)` agar arah regularisasinya sama. MLP tidak memakai bobot kelas (sklearn tidak menyediakannya).""")
book.config("Kepala yang disapu dan grid `C`.", r'''
    C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
    CLF = {"lr":  lambda c: LogisticRegression(C=c, max_iter=2000, class_weight=CLASS_WEIGHT, random_state=SEED),
           "svm": lambda c: LinearSVC(C=c, dual=False, class_weight=CLASS_WEIGHT, random_state=SEED),   # dual=False: lebih cepat saat baris > fitur
           "mlp": lambda c: MLPClassifier((256,), alpha=1 / (100 * c), early_stopping=True, max_iter=300, random_state=SEED)}
''')
book.code(r'''
    rec, val_s, hold_s, fitted, tuning = [], {}, {}, {}, []
    maj = np.bincount(y_train, minlength=C).argmax()
    for S, d in ((val_s, val), (hold_s, holdout)): S["majority"] = np.tile(np.where(np.arange(C) == maj, 0.0, -9.0), (len(d), 1))
    rec.append({"model": "majority", "fit_s": 0.0, "train_f1": mac(y_train, np.full(len(train), maj))})
    for rep, (Xtr, Xva, Xho) in X.items():
        for cname, make in CLF.items():
            t0 = time.perf_counter(); runs = {c: make(c).fit(Xtr, y_train) for c in C_GRID}
            f1 = {c: mac(y_val, m.predict(Xva)) for c, m in runs.items()}; best_c = max(f1, key=f1.get); m = runs[best_c]; name = f"{rep}+{cname}"
            score = (lambda Z: m.decision_function(Z)) if cname == "svm" else (lambda Z: np.log(m.predict_proba(Z) + 1e-9))
            val_s[name], hold_s[name], fitted[name] = score(Xva), score(Xho), m
            tuning += [{"model": name, "C": c, "val_macro_f1": f} for c, f in f1.items()]
            rec.append({"model": name, "best_C": best_c, "fit_s": time.perf_counter() - t0, "train_f1": mac(y_train, m.predict(Xtr))})
    show("Tuning C · macro-F1 validation", pd.DataFrame(tuning).pivot(index="C", columns="model", values="val_macro_f1"))
''')
tail(book, "multiclass_2_pretrained_ekstraksi")
finalize(book, "10_multi_kelas_2_pretrained_ekstraksi.ipynb", "gemastik-mc-2-ekstraksi-fitur", "Gemastik MC 2 Ekstraksi Fitur", gpu=True)
