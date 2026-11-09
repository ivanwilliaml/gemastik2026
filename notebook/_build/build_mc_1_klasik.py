"""Notebook 1 · multiclass dengan fitur klasik: TF-IDF, Word2Vec/FastText, fitur handcrafted + Logistic Regression / Linear SVM."""
from mc_common import Book, head, tail, finalize
from mc_features import features

book = Book()
head(book, "Multiclass 1 · Fitur klasik (TF-IDF, Word2Vec, FastText, fitur handcrafted)", """Tanpa model pretrained. Empat cara mengubah teks jadi angka, ditambah **±75 fitur handcrafted** yang ditempelkan ke TF-IDF lewat ablasi bertahap. Semua dipasangkan dengan Logistic Regression dan Linear SVM.

**Diagram alur, daftar metode, dan kapan memilih apa:** [alur_multi_kelas.html](alur_multi_kelas.html)

| Langkah | Isi |
|---|---|
| 1-2 | Konfigurasi, muat data, audit kualitas |
| 3 | Preprocessing universal (6 view) |
| 4 | Grup duplikat dan split train / validation / holdout |
| 5 | EDA angka |
| 6 | Feature engineering: 6 kelompok fitur, diagnosis MI dan korelasi |
| 7 | Ablasi preprocessing, empat representasi, ablasi fitur E0-E4, sapuan model |
| 8+ | Pilih di validation, post-processing, holdout sekali, analisis kelemahan |

**Data contoh:** [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program), 3 kelas sentimen. Untuk data lain cukup ubah konfigurasi di bagian 1. Jalankan di Kaggle (CPU cukup).""", gpu=False, stem=True)
features(book, 6)

book.md("""## 7. Representasi dan model

Empat representasi, plus satu gabungan TF-IDF + fitur handcrafted, dua pengklasifikasi, grid `C`, dan baseline mayoritas.

| Representasi | Menangkap | Lemah di |
|---|---|---|
| TF-IDF kata (1-2 gram) | kata dan frasa pendek yang khas | tidak tahu kata mirip (`bagus` ≠ `mantap`), kata baru |
| TF-IDF kata + karakter (3-5) | variasi ejaan dan imbuhan | fitur besar, tetap tanpa makna |
| Word2Vec rata-rata | kemiripan makna, dilatih dari train saja | korpus kecil, urutan kata hilang, kata asing = nol |
| FastText rata-rata | seperti Word2Vec + subkata, kata asing tetap dapat vektor | korpus kecil, urutan kata hilang |
| TF-IDF + fitur handcrafted | gaya penulisan, negasi, emoji yang tidak tertangkap kata | fitur redundan menambah noise |

LR memberi probabilitas; SVM memberi margin (tidak bisa dirata-ratakan jadi probabilitas). Semua vectorizer, Word2Vec, dan scaler **hanya belajar dari train**.""")
book.md("### 7.1 Ablasi preprocessing\n\nSatu model (TF-IDF kata + LR) dicoba pada empat view. Kolom `oov_val_%` = persen token validation yang tidak dikenal vocabulary train. View terbaik di validation dipakai untuk TF-IDF kata dan Word2Vec.")
book.config("Parameter TF-IDF kata dan karakter, Word2Vec/FastText, serta grid `C` (kekuatan regularisasi; kecil = lebih kuat).", r'''
    TFIDF_WORD = dict(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)
    W2V = dict(vector_size=100, window=5, min_count=2, sg=1, epochs=20)
    C_GRID = [0.1, 0.3, 1, 3, 10]
''')
book.code(r'''
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_union
    from sklearn.preprocessing import StandardScaler
    from gensim.models import Word2Vec, FastText

    abl = {}
    for v in ["safe", "minimal", "normalized", "clean"]:
        vec = TfidfVectorizer(**TFIDF_WORD).fit(train[v]); lr = LogisticRegression(C=2, max_iter=2000, class_weight=CLASS_WEIGHT).fit(vec.transform(train[v]), y_train)
        abl[v] = {**M(y_val, lr.predict(vec.transform(val[v]))), "fitur": len(vec.vocabulary_),
                  "oov_val_%": 100 * (1 - val[v].str.casefold().str.findall(r"[^\W\d_]+").explode().isin(vec.vocabulary_).mean())}
    show("Ablasi preprocessing · TF-IDF kata + LR · validation", pd.DataFrame(abl).T)
    best_view = max(abl, key=lambda v: abl[v]["macro_f1"]); print("view terbaik untuk representasi kata:", best_view)
''')
book.md("### 7.2 Bangun empat representasi\n\nSetiap representasi menghasilkan `(X_train, X_val, X_holdout)`. TF-IDF kata dan Word2Vec memakai `best_view`; karakter dan FastText memakai `minimal` agar ejaan asli ikut terbaca.")
book.code(r'''
    REPS = {"tfidf_word": best_view, "tfidf_word_char": "minimal", "word2vec": best_view, "fasttext": "minimal"}
    X, rep_vec, splits = {}, {}, (train, val, holdout)

    v = REPS["tfidf_word"]; rep_vec["tfidf_word"] = TfidfVectorizer(**TFIDF_WORD).fit(train[v])
    X["tfidf_word"] = [rep_vec["tfidf_word"].transform(d[v]) for d in splits]

    v = REPS["tfidf_word_char"]; u = make_union(TfidfVectorizer(**TFIDF_WORD), TfidfVectorizer(**TFIDF_CHAR)).fit(train[v])
    X["tfidf_word_char"] = [u.transform(d[v]) for d in splits]

    tok = lambda d, v: d[v].str.casefold().str.findall(r"[^\W\d_]+")                      # token huruf saja
    emb = lambda d, v, m, oov: np.array([np.mean([m.wv[w] for w in q if oov or w in m.wv] or [np.zeros(m.vector_size)], axis=0) for q in tok(d, v)])
    for name, cls, oov in [("word2vec", Word2Vec, False), ("fasttext", FastText, True)]:
        v = REPS[name]; m = cls(tok(train, v).tolist(), **W2V, workers=1, seed=SEED)
        sc = StandardScaler().fit(emb(train, v, m, oov)); X[name] = [sc.transform(emb(d, v, m, oov)) for d in splits]
        known = tok(val, v).explode().dropna().isin(m.wv.index_to_key).mean()
        print(f"{name}: vocab {len(m.wv)} | token validation dikenal {known:.1%} | vektor nol di validation {np.all(emb(val, v, m, False) == 0, axis=1).mean():.1%}")
    show("Bentuk matriks", pd.DataFrame({n: {"train": x[0].shape, "val": x[1].shape, "holdout": x[2].shape} for n, x in X.items()}).T)
''')
book.md("""### 7.3 Ablasi fitur handcrafted (E0-E4)

Fitur ditambahkan **bertahap** ke TF-IDF kata+karakter, bukan sekaligus, supaya terlihat kelompok mana yang benar-benar menambah informasi. `bobot_meta` mengatur seberapa kuat fitur handcrafted dibanding ribuan kolom TF-IDF (karena TF-IDF berskala kecil). Kolom `F1_<kelas>` memperlihatkan apakah fitur membantu kelas minoritas. Setelah itu: **dua uji kegunaan kelompok**, yaitu membuang satu kelompok dari model penuh, dan mengacak satu kelompok di validation (permutation importance, 5 ulangan).""")
book.config("Tahapan ablasi (kumulatif) dan bobot fitur handcrafted yang dicoba.", r'''
    import scipy.sparse as sp
    STAGES = {"E0 tfidf": [], "E1 +structural": ["structural"], "E2 +stylistic +expressive": ["structural", "stylistic", "expressive"],
              "E3 +lexicon": ["structural", "stylistic", "expressive", "lexicon"], "E4 +lexical +discourse": list(GROUP)}
    META_W = [0.1, 0.3, 1.0]
''')
book.code(r'''
    cols_of = lambda groups: sum((GROUP[g] for g in groups), [])
    attach = lambda x, d, cols, w: sp.hstack([x, w * sp.csr_matrix(Zf.loc[d.index, cols].to_numpy())]).tocsr() if cols else x
    rows = []
    for stage, groups in STAGES.items():
        cols = cols_of(groups)
        for w in (META_W if cols else [0]):
            A = [attach(x, d, cols, w) for x, d in zip(X["tfidf_word_char"], splits)]
            for cname, mdl in [("lr", LogisticRegression(C=1, max_iter=2000, class_weight=CLASS_WEIGHT, random_state=SEED)), ("svm", LinearSVC(C=0.3, class_weight=CLASS_WEIGHT, random_state=SEED))]:
                p = mdl.fit(A[0], y_train).predict(A[1])
                rows.append({"tahap": stage, "bobot_meta": w, "model": cname, "n_fitur_meta": len(cols), **M(y_val, p),
                             **{f"F1_{c}": f for c, f in zip(classes, f1_score(y_val, p, labels=range(C), average=None, zero_division=0))}})
    fe_abl = pd.DataFrame(rows); show("Ablasi fitur handcrafted · validation", fe_abl)
    best_fe = fe_abl.sort_values("macro_f1", ascending=False).iloc[0]; FE_GROUPS, FE_W = STAGES[best_fe.tahap], best_fe.bobot_meta
    print("tahap terbaik:", best_fe.tahap, "| bobot:", FE_W, "| macro-F1:", round(best_fe.macro_f1, 4), "| E0:", round(fe_abl[fe_abl.tahap.str.startswith("E0")].macro_f1.max(), 4))
''')
book.code(r'''
    full = cols_of(list(GROUP)); w = FE_W or 1.0
    Afull = [attach(x, d, full, w) for x, d in zip(X["tfidf_word_char"], splits)]
    lr = LogisticRegression(C=1, max_iter=2000, class_weight=CLASS_WEIGHT, random_state=SEED).fit(Afull[0], y_train); base = mac(y_val, lr.predict(Afull[1]))
    drop, perm, rng = {}, {}, np.random.default_rng(SEED)
    for g, cols in GROUP.items():
        keep = [c for c in full if c not in cols]
        A = [attach(x, d, keep, w) for x, d in zip(X["tfidf_word_char"], splits)]
        drop[g] = base - mac(y_val, LogisticRegression(C=1, max_iter=2000, class_weight=CLASS_WEIGHT, random_state=SEED).fit(A[0], y_train).predict(A[1]))
        ds = []
        for _ in range(5):
            Zv = Zf.loc[val.index, full].copy(); Zv[cols] = Zv[cols].sample(frac=1, random_state=int(rng.integers(1e6))).to_numpy()
            ds.append(base - mac(y_val, lr.predict(sp.hstack([X["tfidf_word_char"][1], w * sp.csr_matrix(Zv.to_numpy())]).tocsr())))
        perm[g] = np.mean(ds)
    show("Kegunaan kelompok fitur · selisih macro-F1 (positif = kelompok berguna)", pd.DataFrame({"buang_kelompok": drop, "acak_di_validation": perm, "n_fitur": {g: len(c) for g, c in GROUP.items()}}))
    print("model penuh (semua kelompok):", round(base, 4), "| selisih kecil (< 0,005) berada di dalam noise validation")
''')
book.md("Gabungan TF-IDF + fitur handcrafted terbaik (tahap dan bobot dari tabel di atas) masuk ke sapuan model sebagai representasi tambahan.")
book.code(r'''
    X["tfidf_word_char+meta"] = [attach(x, d, cols_of(FE_GROUPS), FE_W or 1.0) for x, d in zip(X["tfidf_word_char"], splits)]
    print("representasi tambahan:", X["tfidf_word_char+meta"][0].shape, "| kelompok:", FE_GROUPS or "tidak ada (E0 terbaik)")
''')
book.md("""### 7.4 Sapuan model dan parameter

Setiap pasangan (representasi, pengklasifikasi) dicoba pada semua nilai `C`; `C` terbaik dipilih di validation. `train_f1` yang jauh di atas validation menandakan overfit.""")
book.config("Pengklasifikasi yang disapu. Tambahkan model sklearn lain dengan pola `nama: lambda c: Model(...)`.", r'''
    CLF = {"lr":  lambda c: LogisticRegression(C=c, max_iter=2000, class_weight=CLASS_WEIGHT, random_state=SEED),
           "svm": lambda c: LinearSVC(C=c, class_weight=CLASS_WEIGHT, random_state=SEED)}
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
tail(book, "multiclass_1_fitur_klasik", linear_features=True)
finalize(book, "10_multi_kelas_1_fitur_klasik.ipynb", "gemastik-mc-1-fitur-klasik", "Gemastik MC 1 Fitur Klasik", gpu=False)
