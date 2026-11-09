"""Notebook 5 · LightGBM (embedding beku + metadata) + BERT fine-tune + ensemble yang di-tuning dengan OOF."""
from mc_common import Book, head, tail, finalize
from mc_features import features

book = Book()
head(book, "Multiclass 5 · LightGBM + BERT + ensemble yang dituning (OOF)", """Dua jenis model yang saling melengkapi lalu digabung:

- **Cabang A**, model tabular: LightGBM menerima **vektor angka**, bukan teks. Isinya embedding dari encoder pretrained yang dibekukan, ditempel dengan fitur handcrafted.
- **Cabang B**, model kontekstual: BERT yang di-fine-tune pada teks.
- **Ensemble**: bobot dan meta-model dicari dari prediksi *out-of-fold* (OOF), bukan dari prediksi training yang terlalu optimistis. Hyperparameter LightGBM di-tuning dengan random search di atas OOF yang sama.

**Diagram alur, daftar metode, dan kapan memilih apa:** [alur_multi_kelas.html](alur_multi_kelas.html)

| Langkah | Isi |
|---|---|
| 1-5 | Konfigurasi, data, preprocessing, split, EDA (identik dengan notebook lain) |
| 6 | Feature engineering |
| 7 | Embedding beku (satu kali ekstraksi) |
| 8 | Enam cabang model dengan 5-fold OOF: TF-IDF LR, embedding LR, embedding+metadata LR, LightGBM ×2 (tuning), BERT fine-tune |
| 9 | Ensemble: rata-rata sama, bobot dituning, stacking; semuanya dinilai nested |
| 10+ | Pilih, post-processing, holdout sekali, analisis kelemahan |

**Data contoh:** [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program). Jalankan di Kaggle **GPU T4**, internet aktif.""", gpu=True)
features(book, 6)

book.md("""## 7. Embedding beku

Encoder pretrained dibekukan; ekstraksi dilakukan **sekali** untuk seluruh data pengembangan (train + validation) dan holdout, lalu dipakai berulang oleh semua cabang. Mean-pooling dengan attention mask, sehingga padding tidak ikut dihitung. Embedding distandarkan; scaler tidak memakai label.""")
book.config("Encoder beku: (model, awalan, view teks). Ganti dengan `intfloat/multilingual-e5-base` (awalan `query: `, view `embed`) bila hasil notebook 2 menunjukkan e5 lebih kuat.", r'''
    FROZEN = ("indobenchmark/indobert-base-p1", "", "embed")
    MAX_LEN, BATCH = 128, 64
''')
book.code(r'''
    from transformers import AutoTokenizer, AutoModel
    dev = pd.concat([train, val]).sort_index(); y_dev = dev.y.to_numpy()                     # data pengembangan = train + validation
    mid, prefix, view = FROZEN; tk, enc = AutoTokenizer.from_pretrained(mid), AutoModel.from_pretrained(mid).half().cuda().eval(); emb = {}
    for key, d in (("dev", dev), ("hold", holdout)):
        out = []
        for i in range(0, len(d), BATCH):
            b = tk([prefix + s for s in d[view].iloc[i:i + BATCH]], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            with torch.no_grad(): h = enc(**b).last_hidden_state.float()
            msk = b["attention_mask"].unsqueeze(-1); out.append(((h * msk).sum(1) / msk.sum(1)).cpu().numpy())
        emb[key] = np.vstack(out)
    del enc; torch.cuda.empty_cache()
    sc_e = StandardScaler().fit(emb["dev"]); E_dev, E_hold = sc_e.transform(emb["dev"]), sc_e.transform(emb["hold"])
    META = TIER1 + TIER2; Mt_dev, Mt_hold = Zf.loc[dev.index, META].to_numpy(), Zf.loc[holdout.index, META].to_numpy()
    print("data pengembangan:", len(dev), "| holdout:", len(holdout), "| embedding:", E_dev.shape, "| metadata:", Mt_dev.shape)
''')
book.md("""## 8. Cabang model dengan 5-fold OOF

Data pengembangan dibagi lima fold (`StratifiedGroupKFold`, grup duplikat tidak terbelah). Tiap model dilatih lima kali; tiap baris mendapat prediksi dari model yang **tidak pernah melihat baris itu** (OOF). Prediksi holdout = rata-rata log-probabilitas lima model fold. Tanpa refit, jadi holdout tetap bersih.

Dua catatan jujur: (1) epoch BERT **tetap** (`FT_EPOCHS`), tidak dipilih dari fold yang diprediksi, agar OOF tidak bocor; (2) memilih trial LightGBM terbaik dari OOF membuat skor OOF-nya sedikit optimistis, jadi tabel trial menampilkan juga rata-rata dan terburuk semua trial, dan angka paling jujur tetap holdout.""")
book.config("Parameter cabang linear, ruang pencarian LightGBM, dan BERT.", r'''
    K, N_TRIALS = 5, 12
    LR_C = {"tfidf_lr": 3, "emb_lr": 0.05, "emb_meta_lr": 0.05}
    META_W = 1.0                                        # bobot fitur metadata di atas embedding (sudah terstandar)
    SPACE = dict(num_leaves=[7, 15, 31], learning_rate=[.03, .06, .1], n_estimators=[150, 300, 500], min_child_samples=[10, 20, 40],
                 subsample=[.7, .9, 1.0], colsample_bytree=[.2, .4, .7], reg_lambda=[0, 1, 5, 20])
    FT = ("indolem/indobertweet-base-uncased", "bert"); FT_EPOCHS, FT_BATCH, FT_LR, FT_WARMUP, FT_WD = 3, 16, 2e-5, 0.1, 0.01
    TFIDF_WORD = dict(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)
''')
book.code(r'''
    import scipy.sparse as sp, lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_union
    folds = list(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(dev, y_dev, dev.dup_group))
    oof, hold_avg, rec, lp = {}, {}, [], (lambda p: np.log(p + 1e-9))
    show("Fold OOF", pd.DataFrame({f"fold {i}": {"latih": len(tr), "prediksi": len(te)} for i, (tr, te) in enumerate(folds)}).T)

    u = make_union(TfidfVectorizer(**TFIDF_WORD), TfidfVectorizer(**TFIDF_CHAR)).fit(dev.minimal)      # vectorizer tanpa label
    SK = {"tfidf_lr": (u.transform(dev.minimal), u.transform(holdout.minimal)), "emb_lr": (E_dev, E_hold),
          "emb_meta_lr": (np.hstack([E_dev, META_W * Mt_dev]), np.hstack([E_hold, META_W * Mt_hold]))}
    for name, (Xd, Xh) in SK.items():
        t0 = time.perf_counter(); o, h = np.zeros((len(dev), C)), np.zeros((len(holdout), C))
        for tr, te in folds:
            m = LogisticRegression(C=LR_C[name], max_iter=2000, class_weight=CLASS_WEIGHT, random_state=SEED).fit(Xd[tr], y_dev[tr])
            o[te], h = lp(m.predict_proba(Xd[te])), h + lp(m.predict_proba(Xh)) / K
        oof[name], hold_avg[name] = o, h; rec.append({"model": name, "fit_s": time.perf_counter() - t0})
        print(name, "| OOF macro-F1:", round(mac(y_dev, o.argmax(1)), 4))
''')
book.md("### 8.1 LightGBM dengan random search\n\nSetiap trial = satu kombinasi parameter, dinilai dengan 5-fold OOF yang sama. `colsample_bytree` kecil penting karena ada 800 kolom di atas sekitar 3.000 baris. Dua kelompok fitur dibandingkan: embedding saja (E2) dan embedding + metadata (E3).")
book.code(r'''
    rng = np.random.default_rng(SEED); LG = {"emb_lgbm": (E_dev, E_hold), "emb_meta_lgbm": (np.hstack([E_dev, Mt_dev]), np.hstack([E_hold, Mt_hold]))}
    for name, (Xd, Xh) in LG.items():
        t0, trials = time.perf_counter(), []
        for t in range(N_TRIALS):
            p = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}; o, h = np.zeros((len(dev), C)), np.zeros((len(holdout), C))
            for tr, te in folds:
                m = lgb.LGBMClassifier(**p, subsample_freq=1, class_weight=CLASS_WEIGHT, random_state=SEED, n_jobs=4, verbose=-1).fit(Xd[tr], y_dev[tr])
                o[te], h = lp(m.predict_proba(Xd[te])), h + lp(m.predict_proba(Xh)) / K
            trials.append({**p, "oof_macro_f1": mac(y_dev, o.argmax(1)), "_o": o, "_h": h})
        tr_df = pd.DataFrame(trials); best = tr_df.oof_macro_f1.idxmax(); oof[name], hold_avg[name] = trials[best]["_o"], trials[best]["_h"]
        rec.append({"model": name, "fit_s": time.perf_counter() - t0})
        show(f"{name} · 5 trial terbaik", tr_df.drop(columns=["_o", "_h"]).sort_values("oof_macro_f1", ascending=False).head(5))
        print(f"{name} | trial: rata-rata {tr_df.oof_macro_f1.mean():.4f}, terburuk {tr_df.oof_macro_f1.min():.4f}, terbaik {tr_df.oof_macro_f1.max():.4f} | selisih terbaik-rata-rata = optimisme seleksi")
''')
book.md("### 8.2 BERT fine-tune per fold\n\nLoop yang sama dengan notebook 3, tetapi di setiap fold. Jumlah epoch tetap (`FT_EPOCHS`). Bobot kelas dari data pengembangan.")
book.code(r'''
    import gc
    import torch.nn.functional as nnF
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    fmid, fview = FT; tkf = AutoTokenizer.from_pretrained(fmid)
    ids_dev, ids_hold = [tkf(d[fview].tolist(), truncation=True, max_length=MAX_LEN)["input_ids"] for d in (dev, holdout)]
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(C), y=y_dev) if CLASS_WEIGHT == "balanced" else np.ones(C), dtype=torch.float, device="cuda")
    lsm = lambda z: z - np.log(np.exp(z).sum(1, keepdims=True))

    def logits_of(model, ids_):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(ids_), 256): out.append(model(**tkf.pad({"input_ids": ids_[i:i + 256]}, return_tensors="pt").to("cuda")).logits.float().cpu().numpy())
        return np.vstack(out)

    t0, o, h = time.perf_counter(), np.zeros((len(dev), C)), np.zeros((len(holdout), C))
    for k, (tr, te) in enumerate(folds):
        torch.manual_seed(SEED + k); model = AutoModelForSequenceClassification.from_pretrained(fmid, num_labels=C).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WD); steps = FT_EPOCHS * int(np.ceil(len(tr) / FT_BATCH))
        sch, amp = get_linear_schedule_with_warmup(opt, int(FT_WARMUP * steps), steps), torch.amp.GradScaler("cuda")
        for ep in range(FT_EPOCHS):
            model.train(); perm = tr[np.random.RandomState(SEED + ep).permutation(len(tr))]
            for i in range(0, len(perm), FT_BATCH):
                idx = perm[i:i + FT_BATCH]; b = tkf.pad({"input_ids": [ids_dev[j] for j in idx]}, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.float16): loss = nnF.cross_entropy(model(**b).logits.float(), torch.tensor(y_dev[idx], device="cuda"), weight=cw)
                opt.zero_grad(); amp.scale(loss).backward(); amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); amp.step(opt); amp.update(); sch.step()
        o[te], h = lsm(logits_of(model, [ids_dev[j] for j in te])), h + lsm(logits_of(model, ids_hold)) / K
        print(f"fold {k}: macro-F1 {mac(y_dev[te], o[te].argmax(1)):.4f} | {round(time.perf_counter() - t0)} detik")
        del model, opt; gc.collect(); torch.cuda.empty_cache()
    oof["ft_bert"], hold_avg["ft_bert"] = o, h; rec.append({"model": "ft_bert", "fit_s": time.perf_counter() - t0})
''')
book.md("**Ringkasan cabang (OOF).** Tabel konfigurasi E1-E4 dari catatan riset: TF-IDF (E1), embedding + LightGBM (E2), embedding + metadata + LightGBM (E3), BERT fine-tune (E4). Agreement antar cabang menunjukkan seberapa beragam mereka; ensemble hanya berguna bila cabang berbeda pendapat di baris yang berbeda.")
book.code(r'''
    names = list(oof); branch = pd.DataFrame({n: {**M(y_dev, oof[n].argmax(1)), "fit_s": next(r["fit_s"] for r in rec if r["model"] == n)} for n in names}).T
    show("Cabang · OOF pada data pengembangan", branch.sort_values("macro_f1", ascending=False))
    pred = {n: oof[n].argmax(1) for n in names}
    show("Agreement antar cabang · % baris dengan prediksi sama", pd.DataFrame({a: {b: (pred[a] == pred[b]).mean() * 100 for b in names} for a in names}))
    right = np.stack([pred[n] == y_dev for n in names]); print(f"semua cabang benar: {right.all(0).mean():.1%} | minimal satu cabang benar (batas atas ensemble): {right.any(0).mean():.1%} | tidak ada yang benar: {(~right.any(0)).mean():.1%}")
''')
book.md("""## 9. Ensemble yang dituning

Tiga cara menggabungkan cabang, semuanya dinilai **nested** supaya skornya jujur:

| Cara | Cara kerja | Penilaian nested |
|---|---|---|
| `ens_equal` | rata-rata log-probabilitas semua cabang | tidak ada parameter, tidak perlu nested |
| `ens_weighted` | bobot per cabang (kelipatan 0,1, jumlah 1) dicari di OOF untuk memaksimalkan macro-F1 | bobot dicari di empat fold OOF, diterapkan di fold kelima, diulang lima kali |
| `ens_stack` | Logistic Regression di atas log-probabilitas semua cabang | `cross_val_predict` di atas OOF |

Skor `ens_weighted` dan `ens_stack` di bagian evaluasi memakai prediksi nested. Bobot akhir (dari seluruh OOF) hanya dipakai untuk holdout.""")
book.code(r'''
    import itertools
    from sklearn.model_selection import cross_val_predict
    names = list(oof); B = len(names); stack, hstack_ = np.stack([oof[n] for n in names]), np.stack([hold_avg[n] for n in names])
    combos = np.array([c for c in itertools.product(range(11), repeat=B) if sum(c) == 10]) / 10
    mf = lambda y, p: np.mean([2 * np.sum((p == c) & (y == c)) / max(np.sum(p == c) + np.sum(y == c), 1) for c in range(C)])

    def best_w(rows):                                    # bobot terbaik di baris `rows`; seri → log-loss lebih kecil
        S = np.tensordot(combos, stack[:, rows], axes=([1], [0]))                                        # (kombinasi, baris, kelas)
        f1 = np.array([mf(y_dev[rows], s.argmax(1)) for s in S]); ll = -np.take_along_axis(S - np.log(np.exp(S).sum(2, keepdims=True)), y_dev[rows][None, :, None], 2).mean((1, 2))
        return combos[np.lexsort((ll, -f1.round(4)))[0]]

    w_full = best_w(np.arange(len(dev))); nested = np.zeros((len(dev), C))
    for tr, te in folds: nested[te] = np.tensordot(best_w(tr), stack[:, te], axes=([0], [0]))
    show("Bobot ensemble (seluruh OOF)", pd.Series(w_full, index=names).rename("bobot").to_frame())
    insample = mac(y_dev, np.tensordot(w_full, stack, axes=([0], [0])).argmax(1))
    show("Bobot dituning: dalam sampel vs nested", pd.Series({"in_sample_macro_f1": insample, "nested_macro_f1": mac(y_dev, nested.argmax(1)),
                                                              "rata_rata_sama": mac(y_dev, stack.mean(0).argmax(1)), "cabang_terbaik": branch.macro_f1.max()}))
''')
book.code(r'''
    stack_in = lambda a: np.hstack(list(a)); Zst, Zst_h = stack_in(stack), stack_in(hstack_)
    st_oof = cross_val_predict(LogisticRegression(C=1, max_iter=2000, class_weight=CLASS_WEIGHT), Zst, y_dev, cv=folds, method="predict_log_proba")
    st_hold = lp(LogisticRegression(C=1, max_iter=2000, class_weight=CLASS_WEIGHT).fit(Zst, y_dev).predict_proba(Zst_h))
    val_s, hold_s = dict(oof), dict(hold_avg)
    val_s["ens_equal"], hold_s["ens_equal"] = stack.mean(0), hstack_.mean(0)
    val_s["ens_weighted"], hold_s["ens_weighted"] = nested, np.tensordot(w_full, hstack_, axes=([0], [0]))
    val_s["ens_stack"], hold_s["ens_stack"] = st_oof, st_hold
    rec += [{"model": m, "fit_s": 0.0} for m in ("ens_equal", "ens_weighted", "ens_stack")]
    show("Ensemble · OOF (nested untuk weighted dan stack)", pd.DataFrame({m: M(y_dev, val_s[m].argmax(1)) for m in ("ens_equal", "ens_weighted", "ens_stack")}).T)
''')
book.md("""## Evaluasi

Untuk notebook ini, **validation = OOF pada data pengembangan (train + validation)**, bukan satu split validation tunggal. Seleksi model dan bias per kelas memakai OOF itu, holdout dibuka sekali di akhir. Semua cabang dan ensemble tersedia sebagai `val_s` / `hold_s`.""")
book.code(r'''
    val, y_val = dev, y_dev                              # untuk bagian evaluasi: "validation" = OOF data pengembangan
''')
tail(book, "multiclass_5_lgbm_bert_ensemble")
finalize(book, "10_multi_kelas_5_lgbm_bert_ensemble.ipynb", "gemastik-mc-5-lgbm-bert-ensemble", "Gemastik MC 5 LGBM BERT Ensemble", gpu=True)
