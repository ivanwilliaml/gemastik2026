"""Notebook 3 · multiclass dengan fine-tuning penuh tiga model pretrained, loop PyTorch biasa (tanpa Trainer / kelas turunan)."""
from mc_common import Book, head, tail, finalize

book = Book()
head(book, "Multiclass 3 · Pretrained di-fine-tune", """Tiga model pretrained dilatih ulang seluruh bobotnya dengan loss berbobot kelas. Tiap model dicoba pada **dua view teks** (resep asli model vs hasil normalisasi) untuk melihat apakah preprocessing membantu atau malah merusak. Loop pelatihan ditulis langsung dengan PyTorch, bisa dibaca baris demi baris.

**Diagram alur, daftar metode, dan kapan memilih apa:** [alur_multi_kelas.html](alur_multi_kelas.html)

| Langkah | Isi |
|---|---|
| 1-2 | Konfigurasi, muat data, audit kualitas |
| 3 | Preprocessing universal (6 view) |
| 4 | Grup duplikat dan split train / validation / holdout |
| 5 | EDA angka |
| 6 | Panjang token per tokenizer, fine-tuning 3 model × 2 view, kurva belajar |
| 7+ | Pilih di validation, post-processing, holdout sekali, analisis kelemahan |

**Data contoh:** [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program). Jalankan di Kaggle **GPU T4**, internet aktif.""", gpu=True)

book.md("""## 6. Fine-tuning

| Model | Pretraining | View teks yang dibandingkan |
|---|---|---|
| IndoBERTweet | tweet Indonesia | `bert` (resep model card) vs `normalized` |
| IndoBERT p1 | korpus umum Indonesia | `embed` (huruf asli, placeholder) vs `normalized` |
| XLM-RoBERTa base | 100 bahasa | `embed` vs `normalized` |

Resep: AdamW, warmup linear lalu turun linear, fp16 autocast, gradient clipping 1,0, loss cross-entropy dengan bobot kelas dari train. **Checkpoint terbaik = epoch dengan macro-F1 validation tertinggi**; skor holdout pada epoch itu disimpan tetapi tidak dipakai memilih.""")
book.md("### 6.1 Panjang token\n\nTeks yang lebih panjang dari `MAX_LEN` dipotong. Persentase terpotong per tokenizer dan view menentukan apakah `MAX_LEN` cukup.")
book.config("`RUNS` berisi pasangan (model, view). Ubah `EPOCHS`/`LR` bila kurva belajar di 6.3 menunjukkan model belum atau terlalu lama belajar.", r'''
    MODELS = {"indobertweet": "indolem/indobertweet-base-uncased", "indobert_p1": "indobenchmark/indobert-base-p1", "xlm_roberta": "xlm-roberta-base"}
    RUNS = [("indobertweet", "bert"), ("indobertweet", "normalized"), ("indobert_p1", "embed"), ("indobert_p1", "normalized"),
            ("xlm_roberta", "embed"), ("xlm_roberta", "normalized")]
    EPOCHS, MAX_LEN, BATCH, LR, WARMUP, WD = 4, 128, 16, 2e-5, 0.1, 0.01
''')
book.code(r'''
    import gc
    import torch.nn.functional as F
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    toks = {m: AutoTokenizer.from_pretrained(mid) for m, mid in MODELS.items()}
    ln = {(m, v): np.array([len(x) for x in toks[m](train[v].tolist())["input_ids"]]) for m, v in RUNS}
    show("Panjang token train", pd.DataFrame({f"{m}|{v}": {"p50": np.median(a), "p95": np.percentile(a, 95), "p99": np.percentile(a, 99), "max": a.max(),
                                                           f"terpotong_>{MAX_LEN}_%": 100 * (a > MAX_LEN).mean()} for (m, v), a in ln.items()}).T)
''')
book.md("""### 6.2 Fine-tuning

Satu loop untuk semua `RUNS`. Batch dipadatkan per batch (dynamic padding), jadi teks pendek tidak membuang komputasi. Bobot kelas membuat kelas kecil punya pengaruh setara pada loss.""")
book.code(r'''
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(C), y=y_train) if CLASS_WEIGHT == "balanced" else np.ones(C), dtype=torch.float, device="cuda")

    def logits_of(model, tk, ids):                                   # prediksi per batch, urutan baris dipertahankan
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(ids), BATCH * 4): out.append(model(**tk.pad({"input_ids": ids[i:i + BATCH * 4]}, return_tensors="pt").to("cuda")).logits.float().cpu().numpy())
        return np.vstack(out)

    rec, val_s, hold_s, curves = [], {}, {}, []
    for mname, view in RUNS:
        t0, tk, name = time.perf_counter(), toks[mname], f"{mname}|{view}"
        ids = [tk(d[view].tolist(), truncation=True, max_length=MAX_LEN)["input_ids"] for d in (train, val, holdout)]
        torch.manual_seed(SEED); model = AutoModelForSequenceClassification.from_pretrained(MODELS[mname], num_labels=C).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); steps = EPOCHS * int(np.ceil(len(train) / BATCH))
        sch, amp = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps), torch.amp.GradScaler("cuda")
        best_f1 = -1
        for ep in range(EPOCHS):
            model.train(); perm, loss_sum = np.random.RandomState(SEED + ep).permutation(len(train)), 0.0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]; b = tk.pad({"input_ids": [ids[0][j] for j in idx]}, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.float16): loss = F.cross_entropy(model(**b).logits.float(), torch.tensor(y_train[idx], device="cuda"), weight=cw)
                opt.zero_grad(); amp.scale(loss).backward(); amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); amp.step(opt); amp.update(); sch.step()
                loss_sum += loss.item() * len(idx)
            lv = logits_of(model, tk, ids[1]); f1 = mac(y_val, lv.argmax(1))
            curves.append({"model": name, "epoch": ep + 1, "train_loss": loss_sum / len(train), "val_macro_f1": f1})
            if f1 > best_f1: best_f1, val_s[name], hold_s[name] = f1, lv, logits_of(model, tk, ids[2])
        for S in (val_s, hold_s): S[name] = S[name] - np.log(np.exp(S[name]).sum(1, keepdims=True))          # logits → log-probabilitas
        rec.append({"model": name, "fit_s": time.perf_counter() - t0, "best_epoch": int(pd.DataFrame(curves).query("model == @name").val_macro_f1.idxmax() % EPOCHS + 1)})
        print(name, "| macro-F1 validation terbaik:", round(best_f1, 4), "|", round(rec[-1]["fit_s"]), "detik")
        del model, opt; gc.collect(); torch.cuda.empty_cache()
''')
book.md("### 6.3 Kurva belajar dan efek preprocessing\n\n`train_loss` yang terus turun sementara `val_macro_f1` turun berarti overfit. Tabel kedua membandingkan view tiap model: selisih kecil dari noise (lihat CI di evaluasi) tidak bisa dipakai menyimpulkan.")
book.code(r'''
    cv = pd.DataFrame(curves); show("Kurva belajar · macro-F1 validation per epoch", cv.pivot(index="model", columns="epoch", values="val_macro_f1"))
    show("Kurva belajar · train loss per epoch", cv.pivot(index="model", columns="epoch", values="train_loss"))
    pv = pd.DataFrame(rec).assign(model_=lambda d: d.model.str.split("|").str[0], view=lambda d: d.model.str.split("|").str[1]).set_index("model")
    pv["val_macro_f1"] = pd.Series({m: mac(y_val, val_s[m].argmax(1)) for m in val_s})
    show("Efek view teks per model · validation", pv.pivot(index="model_", columns="view", values="val_macro_f1"))
''')
tail(book, "multiclass_3_pretrained_finetune")
finalize(book, "10_multi_kelas_3_pretrained_finetune.ipynb", "gemastik-mc-3-finetune", "Gemastik MC 3 Finetune", gpu=True)
