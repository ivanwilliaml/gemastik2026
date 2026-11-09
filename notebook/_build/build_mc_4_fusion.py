"""Notebook 4 · fine-tune IndoBERT + fusi fitur handcrafted (feature-level fusion di kepala klasifikasi)."""
from mc_common import Book, head, tail, finalize
from mc_features import features

book = Book()
head(book, "Multiclass 4 · BERT fine-tune + fusi metadata", """Pertanyaan notebook ini: **apakah fitur handcrafted menambah informasi yang belum dipelajari BERT?** Lima konfigurasi dengan jumlah fitur bertambah bertahap, dibandingkan pada dua backbone dan dua seed. Vektor kalimat BERT (mean-pooling) ditempel dengan proyeksi fitur metadata, lalu satu kepala klasifikasi belajar dari keduanya.

**Diagram alur, daftar metode, dan kapan memilih apa:** [alur_multi_kelas.html](alur_multi_kelas.html)

| Langkah | Isi |
|---|---|
| 1-2 | Konfigurasi, muat data, audit kualitas |
| 3 | Preprocessing universal (6 view) |
| 4 | Grup duplikat dan split train / validation / holdout |
| 5 | EDA angka |
| 6 | Feature engineering + fitur fragmentasi tokenizer |
| 7 | Fine-tune 2 backbone × 5 konfigurasi × 2 seed, dengan fusi metadata |
| 8+ | Pilih di validation, post-processing, holdout sekali, analisis kelemahan |

**Data contoh:** [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program). Jalankan di Kaggle **GPU T4**, internet aktif.""", gpu=True)
features(book, 6, frag_model="indolem/indobertweet-base-uncased")

book.md("""## 7. Fine-tune dengan fusi metadata

```
teks ──► BERT ──► mean-pool (768) ─────────────┐
                                               ├─► concat ─► dropout ─► linear ─► logit
metadata ─► linear(→64) ─► ReLU ─► dropout ─────┘
```

| Konfigurasi | Fitur metadata di sisi kanan |
|---|---|
| E5 | tidak ada (BERT murni, jalur kode yang sama) |
| E6 | Tier 1 (20 fitur) |
| E7 | Tier 1 + Tier 2 (37 fitur) |
| E8 | Tier 1 + Tier 2 + TF-IDF-SVD (64 komponen) |
| E9 | Tier 1 + Tier 2 + fragmentasi tokenizer (4 fitur) |

BERT dilatih dengan learning rate kecil (`LR`), sedangkan proyeksi metadata dan kepala klasifikasi dengan `LR_HEAD` yang lebih besar karena dimulai dari acak. Dua seed per konfigurasi: selisih antar konfigurasi yang lebih kecil dari selisih antar seed **bukan temuan**. Skor model = rata-rata log-probabilitas dua seed. Checkpoint tiap seed = epoch dengan macro-F1 validation tertinggi.""")
book.config("`BACKBONES`: nama → (model, view teks). `VARIANTS`: konfigurasi → daftar fitur (`svd` = komponen TF-IDF-SVD).", r'''
    BACKBONES = {"indobertweet": ("indolem/indobertweet-base-uncased", "bert"), "indobert_p1": ("indobenchmark/indobert-base-p1", "embed")}
    VARIANTS = {"E5 teks saja": [], "E6 +Tier1": TIER1, "E7 +Tier1+2": TIER1 + TIER2, "E8 +Tier1+2 +SVD": TIER1 + TIER2 + ["svd"],
                "E9 +Tier1+2 +fragmentasi": TIER1 + TIER2 + GROUP["fragmentasi"]}
    SEEDS = [42, 43]
    EPOCHS, MAX_LEN, BATCH, LR, LR_HEAD, WARMUP, WD, META_DIM = 4, 128, 16, 2e-5, 1e-3, 0.1, 0.01, 64
''')
book.code(r'''
    import gc
    import torch.nn.functional as nnF
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoModel, get_linear_schedule_with_warmup

    splits = (train, val, holdout)
    metas = lambda cols: [np.hstack([Zf.loc[d.index, [c for c in cols if c != "svd"]].to_numpy(), Zsvd.loc[d.index].to_numpy() if "svd" in cols else np.empty((len(d), 0))]).astype("float32") for d in splits]
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(C), y=y_train) if CLASS_WEIGHT == "balanced" else np.ones(C), dtype=torch.float, device="cuda")
    show("Ukuran metadata per konfigurasi", pd.DataFrame({v: {"n_fitur": metas(c)[0].shape[1]} for v, c in VARIANTS.items()}).T)
''')
book.md("Dua fungsi pendek dipakai berulang: `fwd` (teks + metadata → logit) dan `predict` (per batch, urutan baris dipertahankan). `enc`, `net`, `head` adalah variabel global yang dibuat ulang tiap run.")
book.code(r'''
    def fwd(b, m):                                       # mean-pool BERT, tempel proyeksi metadata, klasifikasi
        h = enc(**b).last_hidden_state; msk = b["attention_mask"].unsqueeze(-1); z = (h * msk).sum(1) / msk.sum(1)
        return head(torch.cat([z, net(m)], 1) if net is not None else z)

    def predict(ids_, m_):
        enc.eval(); head.eval(); out = []
        if net is not None: net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(ids_), 128):
                b = tk.pad({"input_ids": ids_[i:i + 128]}, return_tensors="pt").to("cuda"); out.append(fwd(b, torch.tensor(m_[i:i + 128], device="cuda")).float().cpu().numpy())
        return np.vstack(out)
''')
book.code(r'''
    rec, val_s, hold_s, per_seed = [], {}, {}, []
    lsm = lambda z: z - np.log(np.exp(z).sum(1, keepdims=True))                         # logits → log-probabilitas
    for bb, (mid, view) in BACKBONES.items():
        tk = AutoTokenizer.from_pretrained(mid); ids = [tk(d[view].tolist(), truncation=True, max_length=MAX_LEN)["input_ids"] for d in splits]
        for vname, cols in VARIANTS.items():
            Mtr, Mva, Mho = metas(cols); name = f"{bb}|{vname}"; vs, hs, t_all = [], [], time.perf_counter()
            for seed in SEEDS:
                torch.manual_seed(seed); enc = AutoModel.from_pretrained(mid).cuda(); md = Mtr.shape[1]
                net = torch.nn.Sequential(torch.nn.Linear(md, META_DIM), torch.nn.ReLU(), torch.nn.Dropout(.1)).cuda() if md else None
                head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(enc.config.hidden_size + (META_DIM if md else 0), C)).cuda()
                small = list(head.parameters()) + (list(net.parameters()) if net is not None else [])
                opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": LR}, {"params": small, "lr": LR_HEAD}], weight_decay=WD)
                steps = EPOCHS * int(np.ceil(len(train) / BATCH)); sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps); amp = torch.amp.GradScaler("cuda")
                best_f1, best_ep = -1, 0
                for ep in range(EPOCHS):
                    enc.train(); head.train(); net is not None and net.train(); perm = np.random.RandomState(seed + ep).permutation(len(train))
                    for i in range(0, len(perm), BATCH):
                        idx = perm[i:i + BATCH]; b = tk.pad({"input_ids": [ids[0][j] for j in idx]}, return_tensors="pt").to("cuda")
                        with torch.autocast("cuda", dtype=torch.float16): loss = nnF.cross_entropy(fwd(b, torch.tensor(Mtr[idx], device="cuda")).float(), torch.tensor(y_train[idx], device="cuda"), weight=cw)
                        opt.zero_grad(); amp.scale(loss).backward(); amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + small, 1.0); amp.step(opt); amp.update(); sch.step()
                    lv = predict(ids[1], Mva); f1 = mac(y_val, lv.argmax(1))
                    if f1 > best_f1: best_f1, best_ep, lbest, hbest = f1, ep + 1, lv, predict(ids[2], Mho)
                vs.append(lsm(lbest)); hs.append(lsm(hbest)); per_seed.append({"backbone": bb, "konfigurasi": vname, "seed": seed, "val_macro_f1": best_f1, "epoch_terbaik": best_ep})
                del enc, net, head, opt; gc.collect(); torch.cuda.empty_cache()
            val_s[name], hold_s[name] = np.mean(vs, axis=0), np.mean(hs, axis=0); rec.append({"model": name, "fit_s": time.perf_counter() - t_all})
            print(name, "| val macro-F1 per seed:", [round(r["val_macro_f1"], 4) for r in per_seed[-len(SEEDS):]], "|", round(rec[-1]["fit_s"]), "detik")
''')
book.md("### 7.1 Apakah metadata membantu?\n\nTabel pertama: rata-rata dan simpangan baku macro-F1 validation antar seed. Tabel kedua: selisih terhadap E5 (teks saja) pada backbone yang sama. Selisih yang lebih kecil dari simpangan baku antar seed tidak bisa disimpulkan; satu seed saja tidak cukup untuk menyatakan sebuah fitur berguna.")
book.code(r'''
    ps = pd.DataFrame(per_seed); agg = ps.groupby(["backbone", "konfigurasi"], sort=False).val_macro_f1.agg(["mean", "std", "min", "max"])
    show("Macro-F1 validation per konfigurasi · antar seed", agg)
    gain = agg["mean"].unstack(0); gain = gain.sub(gain.iloc[0]); show("Selisih terhadap E5 (teks saja) · rata-rata seed", gain)
    show("Epoch terbaik per run", ps.pivot_table(index=["backbone", "konfigurasi"], columns="seed", values="epoch_terbaik", sort=False))
''')
tail(book, "multiclass_4_bert_fusi_metadata")
finalize(book, "10_multi_kelas_4_bert_fusi_metadata.ipynb", "gemastik-mc-4-bert-fusi-metadata", "Gemastik MC 4 BERT Fusi Metadata", gpu=True)
