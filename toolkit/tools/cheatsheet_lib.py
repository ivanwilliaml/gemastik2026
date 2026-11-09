"""Cheatsheet untuk DIKETIK ULANG. Sel literal pendek, tanpa framework runtime."""
from textwrap import dedent


def content():
    cards={}
    def add(key,title,stage,code,needs=(),note='',when='Wajib',mode='inti',replaces='',pre=False):
        cards[key]={'id':key,'title':title,'stage':stage,'code':dedent(code).strip()+'\n',
            'needs':list(needs),'note':note,'when':when,'mode':mode,'replaces':replaces,'pre':pre}
    add('load','Baca data: ubah file dan kolom saja','load','''
        import re, json
        import numpy as np, pandas as pd
        from pathlib import Path
        SEED = 42
        TEXT, LABEL, ID = "text", "label", "id"
        TRAIN, TEST = "train.csv", None  # TEST = "test.csv" bila ada
        df = pd.read_csv(TRAIN)
        te = pd.read_csv(TEST, dtype={ID: str}) if TEST else None
        print(df.shape, df.columns.tolist())
        print(df.head(3).to_string(index=False))
    ''',note='CSV default UTF-8. ID test dibaca sebagai string agar nol di depan tetap utuh. Belum membuang baris apa pun.')
    add('formats','File bukan CSV / encoding berbeda','load','''
        # GANTI satu baris pembacaan; pilih yang sesuai.
        df = pd.read_csv(TRAIN, encoding="latin-1")
        # df = pd.read_csv(TRAIN, sep="\\t")
        # df = pd.read_json(TRAIN, lines=True)
        # df = pd.read_parquet(TRAIN)
        # df = pd.read_excel(TRAIN)
    ''',['load'],note='Jangan mencoba encoding sampai sekadar berhasil tanpa membaca contoh: latin-1 menerima byte apa pun dan bisa menyembunyikan encoding rusak.',mode='ganti',replaces='load')
    add('profile','Missing, unik, kosong, pseudo-null, duplicate','eda','''
        s = df[TEXT].fillna("").astype(str)
        print(pd.DataFrame({"missing": df.isna().sum(),
            "missing_%": df.isna().mean()*100, "unik": df.astype(str).nunique()}))
        blank = s.str.strip().eq("")
        pseudo = s.str.strip().str.lower().isin(["nan", "none", "null", "n/a", "-", "[sticker]"])
        print({"baris": len(df), "kosong": int(blank.sum()), "pseudo_null": int(pseudo.sum()),
            "teks_unik_terisi": s[~blank].nunique(), "duplicate_ekstra": int(s[~blank].duplicated().sum())})
        print(df.sample(min(10, len(df)), random_state=SEED).to_string(index=False))
    ''',['load'],note='Sensus seluruh data. Duplicate ekstra berbeda dari semua anggota duplicate. Pseudo-null dilaporkan dahulu, tidak otomatis menghapus kata valid.')
    add('length','Panjang karakter dan kata','eda','''
        length = pd.DataFrame({"char": s.str.len(), "word": s.str.split().str.len()})
        print(length.describe(percentiles=[.01, .5, .9, .95, .99]).T)
        print("Teks > 1000 karakter (%):", s.str.len().gt(1000).mean()*100)
    ''',['profile'],note='Panjang kata bukan panjang tokenizer BERT. Ukur token model sebelum menentukan max_length.')
    add('vocab','Top words, hapax, bigram/trigram','eda','''
        from collections import Counter
        sample = s.sample(min(5000, len(s)), random_state=SEED)
        words = sample.str.lower().str.findall(r"[^\\W\\d_]+")
        counts = Counter(w for row in words for w in row)
        total, hapax = sum(counts.values()), sum(n == 1 for n in counts.values())
        print({"sampel": len(sample), "token": total, "vocab": len(counts), "hapax": hapax,
            "hapax_%_tipe": 100*hapax/max(len(counts), 1), "hapax_%_token": 100*hapax/max(total, 1)})
        for n in [1, 2, 3]:
            grams = Counter(" ".join(row[i:i+n]) for row in words for i in range(len(row)-n+1))
            print(n, "gram:", grams.most_common(15))
        print("Kandidat aneh:", [(w,n) for w,n in counts.most_common() if len(w)>20 or re.search(r"(.)\\1{2,}",w)][:15])
    ''',['profile'],note='Vocabulary/ngram memakai maksimal 5.000 baris; hapax = frekuensi token satu, bukan document-frequency satu. Kandidat aneh belum tentu typo.')
    add('noise','Jenis noise dan persen masing-masing','eda','''
        patterns = {"url": r"https?://|www\\.", "mention": r"@\\w+", "hashtag": r"#\\w+",
            "angka": r"\\d", "ulang_huruf": r"([^\\W\\d_])\\1{2,}", "ulang_baca": r"[!?.,]{2,}",
            "html": r"</?[a-z][^>]*>", "encoding": r"�|Ã.|Â.", "non_ascii": r"[^\\x00-\\x7f]",
            "emoji": r"[\\U0001F300-\\U0001FAFF]", "invisible": r"[\\u200b\\ufeff]",
            "slang": r"\\b(?:ga|gak|nggak|yg|bgt|utk|udh|klo)\\b", "spasi": r" {2,}"}
        flags = pd.DataFrame({k: s.str.contains(v, regex=True, flags=re.I) for k,v in patterns.items()})
        print(pd.DataFrame({"n": flags.sum(), "%_semua_baris": flags.mean()*100}).round(2))
    ''',['profile'],note='Satu teks bisa mengandung beberapa jenis. Non-ASCII/emoji/slang bukan otomatis noise yang harus dibuang.')
    add('language','Penanda bahasa: coverage heuristik, bukan gold','eda','''
        markers = {"English":r"\\b(?:the|and|this|with)\\b", "Jawa":r"\\b(?:ora|iki|nggih)\\b",
            "Sunda":r"\\b(?:teu|ieu|abdi)\\b"}
        hit = pd.DataFrame({k:s.str.contains(p,case=False,regex=True) for k,p in markers.items()})
        print(pd.DataFrame({"n_hit":hit.sum(), "%_semua_baris":hit.mean()*100}))
        print("Tidak ada penanda (%):",hit.sum(1).eq(0).mean()*100)
        print("Beberapa bahasa terindikasi (%):",hit.sum(1).gt(1).mean()*100)
    ''',['profile'],when='Bahasa campuran / daerah',mode='tambahan',note='Bundle tidak mencatat langid. Ini coverage beberapa kata penanda, BUKAN distribusi bahasa yang sebenarnya. Satu teks dapat kena beberapa; tanpa marker bukan otomatis Indonesia. Audit sampel manusia atau gunakan language-ID yang memang dibawa.')
    add('target','Distribusi kelas dan konflik label','eda','''
        print(df[LABEL].value_counts(dropna=False).to_frame("n"))
        print(df[LABEL].value_counts(normalize=True).mul(100).round(2))
        key = s.str.lower().str.replace(r"\\s+", " ", regex=True).str.strip()
        conflicts = df.groupby(key)[LABEL].nunique().gt(1)
        print("Grup teks dengan label konflik:", int(conflicts.sum()))
        print(df[key.isin(conflicts[conflicts].index)].head(10).to_string(index=False))
    ''',['profile'],note='Jangan ganti konflik ke majority sebelum split. Missing label tidak diubah ke kelas netral.')
    add('clean','Pembersihan minimal untuk classifier','prep','''
        print("Label kosong dikeluarkan dari training:", df[LABEL].isna().sum())
        df = df[df[LABEL].notna()].reset_index(drop=True)
        s = df[TEXT].fillna("").astype(str).str.normalize("NFC")
        s = s.str.replace(r"[\\u200b\\ufeff]", "", regex=True).str.replace(r"\\s+", " ", regex=True).str.strip()
        y = df[LABEL].to_numpy()
        t = te[TEXT].fillna("").astype(str) if te is not None else pd.Series(dtype=str)
        t = t.str.normalize("NFC").str.replace(r"[\\u200b\\ufeff]", "", regex=True).str.replace(r"\\s+", " ", regex=True).str.strip()
    ''',['load'],note='Negasi, emoji, angka dan tanda baca dipertahankan. Teks kosong tetap memiliki baris/label; classifier dapat memakai prior. Test tidak dibuang atau diurutkan ulang.')
    add('normalize','Normalisasi hanya bila diperlukan','prep','''
        slang = {"ga": "tidak", "gak": "tidak", "bgt": "banget"}  # audit dari train
        def normalize(text):
            return re.sub(r"\\b\\w+\\b", lambda m: slang.get(m[0], m[0]), text.lower())
        s = s.map(normalize)
        t = t.map(normalize)
    ''',['clean'],mode='tambahan',when='Slang dominan; bandingkan dengan native',note='Tambahkan sebelum split agar collision normalisasi ikut dikelompokkan. Bukan default untuk semua BERT. Jangan memakai label holdout untuk menulis kamus.')
    add('split','Holdout menurut grup duplicate','split','''
        from sklearn.model_selection import GroupShuffleSplit
        groups = pd.factorize(s.str.lower().str.replace(r"\\s+", " ", regex=True).str.strip())[0]
        assert len(set(groups)) >= 2, "Hanya satu grup: tidak dapat mengukur generalisasi"
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(s, groups=groups))
        assert set(groups[tr]).isdisjoint(groups[va])
        print("Train / valid:", len(tr), len(va))
    ''',['clean'],note='Default aman terhadap exact duplicate. Proporsi kelas tidak dijamin; lihat distribusi train/valid. Untuk campaign/parafrase gunakan ID grup/near-duplicate; untuk shift waktu gunakan split waktu.')
    add('user_groups','Ada penulis/thread/sumber yang harus bersama','split','''
        import networkx as nx
        from sklearn.model_selection import GroupShuffleSplit
        GROUP = "author_id"
        graph = nx.Graph()
        for i, (text, user) in enumerate(zip(s.str.lower(), df[GROUP])):
            graph.add_edge(("row", i), ("text", text))
            if pd.notna(user): graph.add_edge(("row", i), ("user", str(user)))
        owner = {node: k for k, part in enumerate(nx.connected_components(graph)) for node in part}
        groups = np.array([owner[("row", i)] for i in range(len(s))])
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(s, groups=groups))
    ''',['split'],mode='ganti',replaces='split',when='Penulis/thread berulang',note='Menyatukan user dan duplicate; sekadar groupby user masih bisa membocorkan copypasta lintas user. Jalankan import GroupShuffleSplit dari sel split.')
    add('time_split','Test berada pada waktu lebih baru','split','''
        TIME = "published_at"
        time = pd.to_datetime(df[TIME], errors="coerce", utc=True)
        assert time.notna().all(), "Tangani timestamp invalid sebelum split"
        cut = time.quantile(.8)
        tr, va = np.flatnonzero(time < cut), np.flatnonzero(time >= cut)
        assert len(tr) and len(va)
        print("Exact text berulang dari masa lalu:", len(set(s.iloc[tr]) & set(s.iloc[va])))
    ''',['clean'],mode='ganti',replaces='split',when='Evaluasi/test berbasis waktu',note='Tanggal sama tidak dipecah di kedua sisi. Recurrence dilaporkan; jangan memindahkan data masa depan ke train. Untuk waktu lokal tanpa offset, tetapkan timezone sumber dahulu.')
    add('tfidf','TF-IDF kata: baseline yang cepat','feat','''
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=60000, sublinear_tf=True)
        x = v.fit_transform(s.iloc[tr])
        xv = v.transform(s.iloc[va])
        print(x.shape)
    ''',['split'],note='Vocabulary fit hanya di train. Empty vocabulary: input hanya kosong/punctuation/emoji? Pilih character TF-IDF; jangan diam-diam fit pada validation.')
    add('char','Teks pendek, typo, obfuscation: char TF-IDF','feat','''
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(analyzer="char", ngram_range=(1,5), min_df=1, max_features=60000, sublinear_tf=True)
        x, xv = v.fit_transform(s.iloc[tr]), v.transform(s.iloc[va])
    ''',['split'],mode='ganti',replaces='tfidf',note='Mempertahankan emoji dan token satu karakter. Lebih boros daripada word; batasi max_features. Untuk n-gram lebih selektif gunakan char_wb (3,5).')
    add('union','Gabungkan word dan character','feat','''
        from sklearn.pipeline import FeatureUnion
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1,2), max_features=40000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=40000, sublinear_tf=True))])
        x, xv = v.fit_transform(s.iloc[tr]), v.transform(s.iloc[va])
    ''',['split'],mode='ganti',replaces='tfidf',note='Langkah berikutnya setelah baseline; bukan wajib sejak menit pertama. Semua representasi fit train saja.')
    add('lr','Logistic Regression','model','''
        from sklearn.linear_model import LogisticRegression
        assert len(set(y[tr])) >= 2, "Train hanya satu kelas: audit split/data"
        assert set(y[va]) <= set(y[tr]), "Ada kelas valid yang tidak pernah dilihat train"
        m = LogisticRegression(C=2, max_iter=1000, class_weight="balanced")
        m.fit(x, y[tr])
        labels = m.classes_
        p = m.predict_proba(xv)
        pred = labels[p.argmax(1)]
    ''',['tfidf'],note='Label string/angka didukung langsung, tidak perlu encode manual. class_weight balanced adalah kandidat; bandingkan None. Probabilitas selalu mengikuti m.classes_.')
    add('svm','LinearSVC: ganti LR bila fokus F1','model','''
        from sklearn.svm import LinearSVC
        m = LinearSVC(C=1, class_weight="balanced")
        m.fit(x, y[tr])
        pred = m.predict(xv)
    ''',['tfidf'],mode='ganti',replaces='lr',note='Margin SVM bukan probabilitas; jangan sigmoid/softmax lalu menyebutnya terkalibrasi. Sel evaluasi label tetap dapat dipakai; threshold/blend probability LR tidak berlaku.')
    add('dummy','Bandingkan dengan baseline mayoritas','eval','''
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import f1_score
        baseline = DummyClassifier(strategy="most_frequent").fit(x, y[tr])
        print("Dummy macro-F1:", f1_score(y[va], baseline.predict(xv), average="macro", zero_division=0))
    ''',['tfidf'],mode='tambahan',note='Accuracy mayoritas bisa tinggi saat imbalance; lihat macro-F1 dan support.')
    add('evaluate','Evaluasi yang menunjukkan kelemahan','eval','''
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(y[va], pred, labels=np.unique(y), zero_division=0))
        print(pd.DataFrame(confusion_matrix(y[va], pred, labels=np.unique(y)), index=np.unique(y), columns=np.unique(y)))
        errors = df.iloc[va][[TEXT, LABEL]].copy()
        errors["pred"] = pred
        print(errors[errors[LABEL] != errors.pred].head(20).to_string(index=False))
    ''',['lr'],note='Urutan confusion: baris=true, kolom=pred. Baca negasi/sarkasme/sumber dan kelas kecil. Kelas train-only/valid-only dilaporkan, bukan dihilangkan dari macro.')
    add('threshold','Biner: pilih threshold pada bagian tuning','post','''
        from sklearn.metrics import f1_score
        assert len(labels)==2, "Threshold ini khusus dua kelas"
        POS = labels[-1]  # isi label positif sesuai soal, jangan menebak urutannya
        j = list(labels).index(POS)
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(va, groups=groups[va]))
        truth = y[va] == POS
        grid = np.arange(.1, .91, .05)
        cut = max(grid, key=lambda c: f1_score(truth[a], p[a,j]>=c, zero_division=0))
        print("Threshold:", cut, "F1 check:", f1_score(truth[b], p[b,j]>=cut, zero_division=0))
        pred = np.where(p[:,j]>=cut, POS, labels[1-j])
    ''',['lr','split'],mode='tambahan',when='Tepat dua kelas, metrik F1 kelas positif',note='Hanya untuk 2 kelas. Nilai F1 pada bagian tuning a bukan skor jujur; laporkan b. Jangan pakai untuk metrik AUC. Setelah memilih banyak model dengan b, b menjadi development juga.')
    add('class_bias','Multiclass: bias keputusan sederhana','post','''
        from sklearn.metrics import f1_score
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(va, groups=groups[va]))
        weights = np.ones(len(labels))
        for j in range(len(labels)):
            trials = []
            for value in [.5, .75, 1, 1.5, 2]:
                w = weights.copy(); w[j] = value
                trials.append((f1_score(y[va[a]], labels[(p[a]*w).argmax(1)], labels=labels, average="macro", zero_division=0), value))
            weights[j] = max(trials)[1]
        print("Check macro-F1:", f1_score(y[va[b]], labels[(p[b]*weights).argmax(1)], labels=labels, average="macro", zero_division=0))
        pred = labels[(p*weights).argmax(1)]
    ''',['lr','split'],mode='tambahan',note='Gunakan hanya bila lebih baik daripada argmax pada b yang sama. Ini mengubah keputusan, bukan kalibrasi probabilitas. Rare class dan b kecil membuat bias tidak stabil.')
    add('final','Final TF-IDF: refit dan tulis submission','out','''
        from sklearn.base import clone
        assert te is not None, "Isi TEST untuk membuat submission"
        vf, mf = clone(v), clone(m)
        xf = vf.fit_transform(s)
        mf.fit(xf, y)
        result = mf.predict(vf.transform(t))
        submission = pd.DataFrame({ID: te[ID].to_numpy(), LABEL: result})
        assert len(submission) == len(te) and not submission.isna().any().any()
        submission.to_csv("submission.csv", index=False)
    ''',['lr'],note='Format harus mengikuti sample_submission, termasuk ID/urutan/nama kolom. Ini argmax/predict model dasar. Bila threshold/bias terpilih, gunakan sel final post berikut.')
    add('final_post','Submission dengan threshold/bias yang sudah dipilih','out','''
        pf = mf.predict_proba(vf.transform(t))
        assert np.array_equal(mf.classes_, labels)
        # PILIH satu sesuai post yang lolos check:
        result = labels[(pf*weights).argmax(1)]  # multiclass
        # result = np.where(pf[:,j] >= cut, POS, labels[1-j])  # biner
        submission[LABEL] = result
        submission.to_csv("submission.csv", index=False)
    ''',['final'],mode='ganti',replaces='final',note='Tambahkan setelah fit final dan SEBELUM mengirim file. Threshold dari holdout berbeda ukuran train dapat bergeser; OOF lebih stabil bila waktunya cukup.')

    # Task-specific cards are appended below; each remains a literal, typeable cell.
    add_task_cards(add)
    return cards


def add_task_cards(add):
    add('ml_clean','Multilabel: kolom 0/1/unknown','prep','''
        LABELS = ["HS", "Abusive"]  # isi kolom target, bukan kolom metadata
        Y = df[LABELS].apply(pd.to_numeric, errors="raise").to_numpy(float)
        assert np.isin(Y[np.isfinite(Y)], [0,1]).all(), "Label harus 0/1/NaN"
        s = df[TEXT].fillna("").astype(str).str.normalize("NFC").str.strip()
        t = te[TEXT].fillna("").astype(str) if te is not None else pd.Series(dtype=str)
        print(pd.DataFrame({"known": np.isfinite(Y).sum(0), "positive": (Y==1).sum(0),
            "unknown": np.isnan(Y).sum(0)}, index=LABELS))
        complete = np.isfinite(Y).all(1)
        print("Kardinalitas complete:", pd.Series(Y[complete].sum(1)).value_counts().sort_index())
    ''',['load'],note='NaN berbeda dari negatif. Jangan astype(bool), fillna(0), atau argmax. Baris all-zero dapat valid; baris all-unknown tidak berkontribusi pada training.')
    add('ml_list','Label berbentuk JSON list / delimiter','prep','''
        from sklearn.preprocessing import MultiLabelBinarizer
        tags = df[LABEL].map(json.loads)  # contoh: ["marah", "kecewa"]
        # tags = df[LABEL].fillna("").map(lambda x: [w.strip() for w in x.split("|") if w.strip()])
        assert tags.map(lambda x: isinstance(x,list)).all()
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(tags).astype(float)
        LABELS = mlb.classes_.tolist()
        s = df[TEXT].fillna("").astype(str)
        t = te[TEXT].fillna("").astype(str) if te is not None else pd.Series(dtype=str)
    ''',['load'],mode='ganti',replaces='ml_clean',note='Hanya untuk anotasi LENGKAP: label yang tidak tercantum berarti negatif. JSON null/NaN tidak sama dengan list kosong; jangan mengubah missing menjadi [] tanpa ketentuan data.')
    add('ml_eda','Co-occurrence dan density tanpa mengisi unknown','eda','''
        known = np.isfinite(Y)
        positive = (Y==1).astype(int)
        both_known = known.astype(int).T @ known.astype(int)
        co = positive.T @ positive
        print(pd.DataFrame(np.divide(co*100., both_known, out=np.full(co.shape, np.nan), where=both_known>0), index=LABELS, columns=LABELS))
        print("Density complete:", Y[known.all(1)].mean(), "n_complete:", known.all(1).sum())
    ''',['ml_clean'],note='Penyebut co-occurrence adalah pasangan label diketahui. Sampel tanpa label lengkap tidak ikut density complete.')
    add('ml_model','Binary relevance LR, unknown dimasking','model','''
        from sklearn.linear_model import LogisticRegression
        heads, p = [], np.zeros((len(va), Y.shape[1]))
        for j in range(Y.shape[1]):
            k = np.isfinite(Y[tr,j]); values = np.unique(Y[tr,j][k])
            if len(values)<2:
                heads.append(float(values[0]) if len(values) else .5)
                p[:,j] = heads[-1]
            else:
                m = LogisticRegression(C=2, max_iter=1000, class_weight="balanced").fit(x[k], Y[tr,j][k])
                heads.append(m); p[:,j] = m.predict_proba(xv)[:,list(m.classes_).index(1)]
        pred = p >= .5
        check = np.arange(len(va))
    ''',['tfidf','ml_clean'],note='Label train konstan memakai prediksi konstan; all-unknown memakai prior .5 dan harus dilaporkan sebagai belum dipelajari. Setiap label dilatih hanya pada observasi diketahui.')
    add('ml_eval','F1, support, false positive / negative per label','eval','''
        from sklearn.metrics import precision_recall_fscore_support, average_precision_score
        rows = []
        for j, label in enumerate(LABELS):
            k = np.isfinite(Y[va[check],j]); truth = Y[va[check],j][k]; guess = pred[check,j][k]
            pr, rc, f1, _ = precision_recall_fscore_support(truth, guess, average="binary", zero_division=0) if k.any() else (np.nan,)*4
            ap = average_precision_score(truth, p[check,j][k]) if (truth==1).any() else np.nan
            rows.append([label, k.sum(), (truth==1).sum(), pr, rc, f1, ap,
                ((truth==0)&guess).sum(), ((truth==1)&~guess).sum()])
        report = pd.DataFrame(rows, columns=["label","known","positive","precision","recall","f1","AP","FP","FN"])
        print(report.to_string(index=False)); print("macro-F1:", report.f1.mean())
    ''',['ml_model'],note='AP dari probabilitas p, bukan keputusan 0/1. Label tanpa positif: AP undefined (NaN). Exact match hanya masuk akal pada baris label lengkap.')
    add('ml_threshold','Threshold per label: tune/check terpisah','post','''
        from sklearn.metrics import f1_score
        a,b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(va, groups=groups[va]))
        cuts = np.full(Y.shape[1], .5)
        for j in range(Y.shape[1]):
            k = np.isfinite(Y[va[a],j]); truth = Y[va[a],j][k]
            if min((truth==1).sum(), (truth==0).sum()) < 5: continue
            cuts[j] = max(np.arange(.1,.91,.05), key=lambda c: f1_score(truth, p[a,j][k]>=c, zero_division=0))
        pred = p >= cuts
        check = b
        print("Threshold:", dict(zip(LABELS, cuts)))
        print("Jalankan ulang ml_eval: check=b sudah mengecualikan baris tuning")
    ''',['ml_model','split'],mode='tambahan',note='Pada sel evaluasi gunakan indeks check b, bukan semua validation yang ikut tuning. Rare label fallback .5. Jangan memaksa minimal satu label aktif.')
    add('ml_final','Refit binary relevance dan submission kolom label','out','''
        from sklearn.linear_model import LogisticRegression
        xf = v.fit_transform(s); xt = v.transform(t)
        pf = np.zeros((len(t), Y.shape[1]))
        for j in range(Y.shape[1]):
            k = np.isfinite(Y[:,j]); values = np.unique(Y[k,j])
            if len(values)<2: pf[:,j] = values[0] if len(values) else .5
            else:
                m = LogisticRegression(C=2, max_iter=1000, class_weight="balanced").fit(xf[k], Y[k,j])
                pf[:,j] = m.predict_proba(xt)[:,list(m.classes_).index(1)]
        submission = pd.DataFrame((pf>=.5).astype(int), columns=LABELS)
        submission.insert(0, ID, te[ID].to_numpy())
        submission.to_csv("submission.csv", index=False)
    ''',['ml_model'],mode='tambahan',note='Memerlukan TEST. Bila threshold lolos check, ganti .5 dengan cuts. Jika format jawaban list label, ubah format sesuai sample_submission; urutan baris tetap.')
    add('absa_clean','ABSA: satu kolom kategorikal per aspek','prep','''
        ASPECTS = ["menu", "distribusi"]
        # NaN = unknown; "tidak_ada" boleh dipakai hanya bila anotasinya eksplisit.
        s = df[TEXT].fillna("").astype(str)
        print(df[ASPECTS].apply(lambda c: c.value_counts(dropna=False)))
        print("Known per aspek:", df[ASPECTS].notna().sum().to_dict())
    ''',['load'],note='ABSA dengan label positif/netral/negatif bukan matriks multilabel 0/1. Missing tidak otomatis berarti aspek tidak ada. Split pada level komentar sebelum mengubah menjadi pasangan aspek-teks.')
    add('absa_model','Satu classifier per aspek dengan split yang sama','model','''
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        aspect_models = {}
        for aspect in ASPECTS:
            k = df.iloc[tr][aspect].notna().to_numpy()
            truth = df.iloc[tr][aspect].to_numpy()[k]
            if len(np.unique(truth))<2: print(aspect, "train tidak cukup kelas"); continue
            m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x[k], truth)
            aspect_models[aspect] = m
            z = df.iloc[va][aspect].notna().to_numpy()
            if z.any(): print(aspect, classification_report(df.iloc[va][aspect].to_numpy()[z], m.predict(xv[z]), zero_division=0))
    ''',['tfidf','absa_clean'],note='Evaluasi per aspek dan support. Unknown dikecualikan. Model satu kelas dapat memakai DummyClassifier, tetapi itu belum mempelajari pembeda kelas.')
    add('unsup_clean','Tanpa label: simpan teks asli dan volume','prep','''
        s = df[TEXT].fillna("").astype(str).str.normalize("NFC").str.strip()
        keep = s.ne("")
        df, s = df[keep].reset_index(drop=True), s[keep].reset_index(drop=True)
        print("Teks kosong dibuang:", int((~keep).sum()))
    ''',['load'],note='Tidak dedup seluruh corpus secara diam-diam: jumlah komentar berulang bisa penting untuk tren. Fit topik memakai teks unik train; assignment mempertahankan volume.')
    add('topic_model','NMF topik: fit train saja','model','''
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        K = 6  # harus lebih kecil daripada jumlah dokumen/vocabulary train
        fit_text = s.iloc[tr].drop_duplicates()
        v = TfidfVectorizer(min_df=1, max_features=30000)
        x = v.fit_transform(fit_text)
        assert min(x.shape)>K, "Kurangi K atau tambah teks unik"
        m = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=SEED).fit(x)
        topics = m.transform(v.transform(s)).argmax(1)
        terms = np.array(v.get_feature_names_out())
        for j,row in enumerate(m.components_): print(j, terms[row.argsort()[-10:][::-1]])
    ''',['unsup_clean','split'],note='Pilih K dari validation dan baca contoh tiap topik. Jangan fit ulang setiap periode: ID topik akan bergeser. Semua-zero/OOV cenderung masuk topik 0; audit coverage.')
    add('lda','LDA: gunakan counts, bukan TF-IDF','model','''
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        v = CountVectorizer(min_df=1, max_features=30000)
        x = v.fit_transform(s.iloc[tr].drop_duplicates())
        m = LatentDirichletAllocation(n_components=K, max_iter=10, random_state=SEED).fit(x)
        topics = m.transform(v.transform(s)).argmax(1)
        terms = np.array(v.get_feature_names_out())
        for j,row in enumerate(m.components_): print(j, terms[row.argsort()[-10:][::-1]])
    ''',['topic_model'],mode='ganti',replaces='topic_model',note='K berasal dari konfigurasi. Counts diperlukan untuk model LDA; jangan menamakan distribusi topik sebagai probabilitas kebenaran.')
    add('cluster','TF-IDF → SVD → KMeans','model','''
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        from sklearn.cluster import MiniBatchKMeans
        K = 6
        dim = min(100, min(x.shape)-1)
        assert dim>=2 and x.shape[0]>K, "Kurangi K / tambah data"
        svd = TruncatedSVD(n_components=dim, random_state=SEED).fit(x)
        z, zv = normalize(svd.transform(x)), normalize(svd.transform(xv))
        m = MiniBatchKMeans(n_clusters=K, n_init=10, random_state=SEED).fit(z)
        topics = m.predict(normalize(svd.transform(v.transform(s))))
        valid_clusters = m.predict(zv)
    ''',['tfidf'],note='SVD wajib fit train saja; dim dibatasi ukuran data. ID cluster arbitrer. Untuk plot 2D, jangan menganggap geometri gambar sebagai kualitas cluster.')
    add('cluster_eval','Ukuran, contoh, silhouette dan vocabulary coverage','eval','''
        from sklearn.metrics import silhouette_score
        print(pd.Series(topics).value_counts().sort_index())
        table = df[[TEXT]].assign(cluster=topics)
        print(table.groupby("cluster").head(5).to_string(index=False))
        if 1<len(set(valid_clusters))<len(valid_clusters):
            print("Silhouette valid:", silhouette_score(zv, valid_clusters, sample_size=min(1000,len(zv)), random_state=SEED))
        print("Validation zero-vector (%):", np.asarray(xv.getnnz(axis=1)==0).mean()*100)
    ''',['cluster'],note='Silhouette pada sampel, bukan matriks jarak penuh. Bandingkan K dengan data/representasi sama. Jangan menyebut silhouette sebagai akurasi.')
    add('topic_eval','Contoh topik dan coverage, tanpa klaim accuracy','eval','''
        table = df[[TEXT]].assign(topic=topics)
        print(table.groupby("topic").size())
        print(table.groupby("topic").head(5).to_string(index=False))
        valid_x = v.transform(s.iloc[va])
        print("Validation OOV/zero-vector (%):", np.asarray(valid_x.getnnz(axis=1)==0).mean()*100)
        pd.DataFrame({"topic": topics}).to_csv("topics.csv", index=False)
    ''',['topic_model'],note='Audit nama topik, top words, dokumen tidak cocok dan stabilitas seed. NPMI/diversity dapat ditambahkan bila ada waktu; contoh manusia tetap diperlukan.')
    add('temporal','Tren: hitungan dan PROPORSI topik','eval','''
        TIME = "published_at"
        time = pd.to_datetime(df[TIME], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta")
        period = time.dt.tz_localize(None).dt.to_period("W").dt.start_time
        counts = pd.crosstab(period, topics)
        shares = counts.div(counts.sum(axis=1), axis=0)
        print(counts.sum(axis=1).rename("n")); print(shares.round(3))
        print("Support kecil:", counts.sum(axis=1).lt(30))
        shares.to_csv("topic_share.csv")
    ''',['topic_model'],note='Gunakan time_split sebelum fit. utc=True menganggap timestamp tanpa zona adalah UTC; ubah bila sumbernya WIB. Missing tanggal harus ditangani. Volume naik berbeda dari proporsi isu naik.')
    add('reg_clean','Target numerik: validasi skala, outlier, missing','prep','''
        values = pd.to_numeric(df[LABEL], errors="coerce")
        print("Target invalid/missing:", (~np.isfinite(values)).sum())
        df = df[np.isfinite(values)].copy().reset_index(drop=True)
        y = pd.to_numeric(df[LABEL]).to_numpy(float)
        s = df[TEXT].fillna("").astype(str)
        t = te[TEXT].fillna("").astype(str) if te is not None else pd.Series(dtype=str)
        print(pd.Series(y).describe(percentiles=[.01,.5,.95,.99]))
    ''',['load'],note='Target biner float tidak otomatis regresi. Tentukan dari soal. Jangan clip berdasarkan nilai target test atau mengubah rating menjadi nominal tanpa mempertimbangkan ordinal.')
    add('ridge','Ridge untuk regresi teks','model','''
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        m = Ridge(alpha=5).fit(x, y[tr])
        pred = m.predict(xv)
        print("MAE:", mean_absolute_error(y[va], pred), "RMSE:", root_mean_squared_error(y[va], pred))
        print("Mean baseline RMSE:", root_mean_squared_error(y[va], np.full(len(va), y[tr].mean())))
        print(df.iloc[va][[TEXT,LABEL]].assign(pred=pred, error=np.abs(y[va]-pred)).nlargest(15,"error"))
    ''',['tfidf','reg_clean'],note='Untuk MAE gunakan median baseline. Prediksi mean optimal terhadap MSE/RMSE. Jika output ordinal, QWK/cutpoint merupakan task berbeda.')
    add('pair_clean','Dua teks: pertahankan arah dan label asli','prep','''
        A, B = "premise", "hypothesis"
        TEXT = A
        df = df[df[LABEL].notna()].reset_index(drop=True)
        a = df[A].fillna("").astype(str).str.normalize("NFC").str.strip()
        b = df[B].fillna("").astype(str).str.normalize("NFC").str.strip()
        s, y = a, df[LABEL].to_numpy()
        print("Pasangan identik:", a.eq(b).sum(), "kosong kiri/kanan:", a.eq("").sum(), b.eq("").sum())
        print(df[LABEL].value_counts(dropna=False))
    ''',['load'],note='NLI dapat tiga kelas entailment/neutral/contradiction. Parafrase/duplicate/matching dapat dua kelas. Jangan memaksa semua task pasangan menjadi NLI, jangan mengasumsikan ID 0/1/2 artinya sama.')
    add('pair_split','Split connected endpoints pasangan','split','''
        import networkx as nx
        from sklearn.model_selection import GroupShuffleSplit
        graph = nx.Graph()
        graph.add_edges_from(zip(a.str.lower(), b.str.lower()))
        owner = {node:k for k,part in enumerate(nx.connected_components(graph)) for node in part}
        groups = a.str.lower().map(owner).to_numpy()
        assert len(set(groups))>=2, "Semua pair terhubung; perlu split resmi/komponen independen"
        tr,va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(a, groups=groups))
        assert not (set(a.iloc[tr].str.lower()) | set(b.iloc[tr].str.lower())) & (set(a.iloc[va].str.lower()) | set(b.iloc[va].str.lower()))
    ''',['pair_clean'],note='Teks sama di sisi manapun tidak boleh muncul di kedua split. Buang/tangani sisi kosong sebelum grouping agar semua missing tidak membentuk komponen raksasa.')
    add('pair_features','TF-IDF relasi, bukan sekadar gabungan kalimat','feat','''
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        v = TfidfVectorizer(ngram_range=(1,2), max_features=40000, sublinear_tf=True)
        v.fit(pd.concat([a.iloc[tr], b.iloc[tr]]))
        def features(indices):
            u,w = v.transform(a.iloc[indices]), v.transform(b.iloc[indices])
            return hstack([u, w, abs(u-w), u.multiply(w)], format="csr")
        x, xv = features(tr), features(va)
    ''',['pair_split'],note='Arah u,w bermakna pada NLI. Untuk STS/parafrase simetris: hapus u,w dari hstack dan gunakan abs(u-w), u*w saja.')
    add('sts_clean','STS: target kontinu, bukan label kategori','prep','''
        A, B = "text_1", "text_2"
        TEXT = A
        df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
        df = df[np.isfinite(df[LABEL])].reset_index(drop=True)
        a,b = df[A].fillna("").astype(str), df[B].fillna("").astype(str)
        s,y = a,df[LABEL].to_numpy(float)
        print(pd.Series(y).describe())
    ''',['load'],note='Isi LABEL sesuai kolom skor. Rentang skor ditentukan dari task, bukan diasumsikan semua data 0–5.')
    add('sts_model','STS simetris + Ridge + korelasi','model','''
        from scipy.sparse import hstack
        from sklearn.linear_model import Ridge
        from scipy.stats import pearsonr, spearmanr
        def features(indices):
            u,w = v.transform(a.iloc[indices]), v.transform(b.iloc[indices])
            return hstack([abs(u-w), u.multiply(w)], format="csr")
        x,xv = features(tr),features(va)
        m = Ridge(alpha=5).fit(x,y[tr]); pred = m.predict(xv)
        print("Pearson:", pearsonr(y[va],pred).statistic, "Spearman:", spearmanr(y[va],pred).statistic)
    ''',['pair_features','sts_clean'],note='Konstan/sampel satu membuat korelasi undefined. Tambahkan RMSE/MAE untuk kesalahan skala. Jangan train ID baris sebagai skor (dataset STS bisa punya score ID dan correlation target berbeda).')
    add('embed','Embedding E5 dari folder offline','feat','''
        import torch
        from transformers import AutoTokenizer, AutoModel
        MODEL = "C:/gemastik/models/e5-small"
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        enc = AutoModel.from_pretrained(MODEL, local_files_only=True).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = enc.to(device)
        @torch.inference_mode()
        def embed(texts, role="query"):
            rows = []
            for i in range(0,len(texts),16):
                batch = tok([role+": "+str(t) for t in list(texts)[i:i+16]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                h = enc(**batch).last_hidden_state; mask = batch.attention_mask.unsqueeze(-1)
                z = (h*mask).sum(1)/mask.sum(1).clamp(min=1)
                rows.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
            return np.vstack(rows)
    ''',['load'],mode='tambahan',pre=True,note='Tidak mengunduh. Isi path folder yang benar. Query: untuk klasifikasi/STS/kedua sisi; passage: untuk dokumen retrieval. Encoder beku dapat dipakai semua split, tetapi head hanya fit train.')
    add('embed_head','LR di atas embedding beku','model','''
        from sklearn.linear_model import LogisticRegression
        x, xv = embed(s.iloc[tr].tolist()), embed(s.iloc[va].tolist())
        m = LogisticRegression(C=1, max_iter=1000, class_weight="balanced").fit(x,y[tr])
        labels = m.classes_; p = m.predict_proba(xv); pred = labels[p.argmax(1)]
    ''',['embed','split'],mode='ganti',replaces='tfidf',pre=True,note='Mengganti representasi DAN sel LR. Sel evaluasi label sama. Final inference harus memakai embed(t), bukan vf.transform(t).')
    add('bert_setup','BERT: path lokal, mapping label, dataset pendek','model','''
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
        MODEL = "C:/gemastik/models/indobertweet"
        labels = np.unique(y[tr]); lookup = {v:i for i,v in enumerate(labels)}
        assert set(y[va]) <= set(labels), "Ada label valid yang tidak ada di train"
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        net = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True,
            num_labels=len(labels), ignore_mismatched_sizes=True)
        def dataset(indices, with_labels=True):
            batch = tok(s.iloc[indices].tolist(), truncation=True, max_length=192)
            return [{**{k:v[i] for k,v in batch.items()}, **({"labels": lookup[y[j]]} if with_labels else {})}
                for i,j in enumerate(indices)]
    ''',['split'],mode='tambahan',pre=True,note='Hanya single-label classifier. IndoBERTweet: pakai sel tweet_prep dahulu. IndoBERT P1/XLM-R/NusaBERT native sebagai pembanding; ubah MODEL saja bila arsitektur klasifikasi cocok.')
    add('tweet_prep','Preprocessing khusus IndoBERTweet','prep','''
        import emoji
        def tweet(text):
            text = re.sub(r"https?://\\S+|www\\.\\S+", "HTTPURL", text.lower())
            text = re.sub(r"(?<!\\w)@\\w+", "@USER", text)
            return emoji.demojize(text)
        s, t = s.map(tweet), t.map(tweet)
    ''',['clean'],mode='tambahan',pre=True,note='Jalankan sebelum split supaya collision setelah normalisasi ikut group. Jangan memakai ini untuk offset QA/NER. Literal placeholder USER/URL dari dataset perlu diperiksa sendiri.')
    add('bert_train','Fine-tuning: satu holdout, tanpa CV berat','model','''
        from sklearn.metrics import f1_score
        args = TrainingArguments(output_dir="bert_run", num_train_epochs=2, learning_rate=2e-5,
            per_device_train_batch_size=8, per_device_eval_batch_size=16,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
            fp16=torch.cuda.is_available(), report_to="none", seed=SEED)
        trainer = Trainer(model=net, args=args, train_dataset=dataset(tr), eval_dataset=dataset(va),
            data_collator=DataCollatorWithPadding(tok),
            compute_metrics=lambda o: {"f1": f1_score(o.label_ids, o.predictions.argmax(1), average="macro", zero_division=0)})
        trainer.train()
        from scipy.special import softmax
        p = softmax(trainer.predict(dataset(va,False)).predictions, axis=1)
        pred = labels[p.argmax(1)]
    ''',['bert_setup'],mode='tambahan',pre=True,note='Label kolom konsisten lewat lookup. Skor adalah development karena checkpoint dipilih di validation. fp16 untuk CUDA; jangan mengaktifkan bf16 pada RTX 3070 tanpa dukungan. Turunkan batch/length jika OOM.')
    add('bert_predict','Prediksi test dari BERT yang dilatih','out','''
        batch = tok(t.tolist(), truncation=True, max_length=192)
        test_data = [{k:v[i] for k,v in batch.items()} for i in range(len(t))]
        pt = softmax(trainer.predict(test_data).predictions, axis=1)
        submission = pd.DataFrame({ID: te[ID].to_numpy(), LABEL: labels[pt.argmax(1)]})
        submission.to_csv("submission.csv", index=False)
        trainer.save_model("model_final"); tok.save_pretrained("model_final")
    ''',['bert_train'],mode='tambahan',pre=True,note='Ini model holdout, bukan refit seluruh train. Bila refit, epoch dibekukan dari validation; jangan menyetel epoch dengan test. Untuk ensemble, simpan probability dan labels sebelum menimpa variabel.')
    add('token_length','Persentil token dari tokenizer yang dipakai','eda','''
        sample = s.sample(min(2000,len(s)), random_state=SEED).tolist()
        lengths = pd.Series([len(row) for row in tok(sample, truncation=False)["input_ids"]])
        print(lengths.quantile([.5,.9,.95,.99])); print("Terpotong @192 (%):", lengths.gt(192).mean()*100)
    ''',['bert_setup'],mode='tambahan',pre=True,note='Untuk pair, tokenize dua sisi bersamaan. QA perlu stride/offset, bukan sekadar memperbesar max_length.')
    add('ensemble','Rata-rata probability: label dan baris wajib sama','post','''
        # Simpan setiap kandidat SEBELUM variabel p ditimpa.
        # p_lr, p_bert; test_lr, test_bert; labels_lr, labels_bert
        assert np.array_equal(labels_lr, labels_bert)
        assert p_lr.shape == p_bert.shape and test_lr.shape == test_bert.shape
        blend_valid = (p_lr + p_bert)/2
        blend_test = (test_lr + test_bert)/2
        guess = labels_lr[blend_valid.argmax(1)]
    ''',['lr'],mode='tambahan',note='Prediksi harus pada indeks validation/test yang SAMA. Bukan mencampur skor OOF dan holdout yang berbeda. Mulai bobot 50:50; optimasi bobot/bias pada subset tuning saja.')
    add('calibrate','Kalibrasi scalar temperature, bila metrik probabilitas','post','''
        from scipy.special import softmax
        from scipy.optimize import minimize_scalar
        from sklearn.metrics import log_loss
        a,b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(va, groups=groups[va]))
        logits = np.log(np.clip(p, 1e-7, 1))
        loss = lambda T: log_loss(y[va[a]], softmax(logits[a]/T,axis=1), labels=labels)
        temperature = minimize_scalar(loss, bounds=(.3,4), method="bounded").x
        print("Check log-loss:", log_loss(y[va[b]], softmax(logits[b]/temperature,axis=1), labels=labels))
    ''',['lr','split'],mode='tambahan',note='Temperature tidak mengubah argmax sehingga bukan teknik menaikkan F1 label langsung. Bandingkan log-loss raw pada b. Label order harus diurut sesuai kolom p.')
    add('word2vec','Word2Vec dari train sendiri','feat','''
        from gensim.models import Word2Vec
        tokens = s.str.lower().str.findall(r"[^\\W\\d_]+")
        w2v = Word2Vec(tokens.iloc[tr].tolist(), vector_size=100, min_count=1, workers=1, seed=SEED)
        def vectors(rows):
            return np.array([np.mean([w2v.wv[w] for w in row if w in w2v.wv],axis=0)
                if any(w in w2v.wv for w in row) else np.zeros(100) for row in rows])
        x,xv = vectors(tokens.iloc[tr]), vectors(tokens.iloc[va])
    ''',['split'],mode='ganti',replaces='tfidf',note='Tidak butuh pretrained. Fit korpus train saja. W2V kecil bukan jaminan lebih baik daripada TF-IDF; semua-OOV diberi vektor nol dan harus dihitung.')
    add('retrieval_load','Corpus dan query dengan banyak jawaban relevan','load','''
        import json, re
        import numpy as np, pandas as pd
        SEED = 42
        corpus = pd.read_csv("corpus.csv", dtype={"doc_id":str})
        queries = pd.read_csv("queries.csv")
        docs = corpus["text"].fillna("").astype(str).tolist()
        ids = corpus.doc_id.astype(str).to_numpy()
        assert corpus.doc_id.notna().all() and len(set(ids))==len(ids)
        gold = [set(map(str,json.loads(v))) for v in queries.relevant_ids]
        assert all(g and g<=set(ids) for g in gold), "Gold ID tidak ada di corpus / kosong"
    ''',note='relevant_ids berupa JSON list, contoh ["001","002"]. Jangan membuat negatif dari dokumen yang belum dinilai. Corpus index bukan supervised training split.')
    add('retrieval_split','Holdout per query yang sama','split','''
        from sklearn.model_selection import GroupShuffleSplit
        q = queries["query"].fillna("").astype(str)
        groups = pd.factorize(q.str.lower().str.strip())[0]
        assert q.str.strip().ne("").all()
        tr,va = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(q, groups=groups))
    ''',['retrieval_load'],note='tr = query tuning, va = query check. Jangan split baris pasangan query-doc secara random. Kueri ringkasan-artikel hanya proxy dan biasanya terlalu mudah.')
    add('bm25','BM25: scoring satu query sekali','model','''
        from sklearn.feature_extraction.text import CountVectorizer
        v = CountVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
        matrix = v.fit_transform(docs).astype(float).tocsr()
        n, length = len(docs), np.asarray(matrix.sum(1)).ravel()
        freq = np.asarray((matrix>0).sum(0)).ravel()
        idf = np.log(1+(n-freq+.5)/(freq+.5))
        rows = np.repeat(np.arange(n), np.diff(matrix.indptr))
        tf = matrix.data.copy()
        matrix.data = idf[matrix.indices]*tf*2.5/(tf+1.5*(.25+.75*length[rows]/length.mean()))
        ranked = []
        for query in q:
            score = ((v.transform([query])>0) @ matrix.T).toarray().ravel()
            ranked.append(np.argsort(-score,kind="stable")[:10])
        print([[ids[i] for i in row[:3]] for row in ranked[:3]])
    ''',['retrieval_split'],note='Tanpa paket rank-bm25 yang tidak tercatat di bundle. k1=1.5, b=.75, query term binary. Tidak membuat matriks query×corpus penuh. Doc ID tetap string.')
    add('retrieval_eval','MRR@K, Recall@K, NDCG@K multi-relevance','eval','''
        K = 10
        rows = []
        for i in va:
            found = [ids[j] for j in ranked[i][:K]]
            hit = np.array([doc in gold[i] for doc in found], dtype=float)
            pos = np.flatnonzero(hit)
            ideal = np.sum(1/np.log2(np.arange(min(K,len(gold[i])))+2))
            rows.append([1/(pos[0]+1) if len(pos) else 0, len(set(found)&gold[i])/len(gold[i]),
                np.sum(hit/np.log2(np.arange(len(hit))+2))/ideal])
        print(pd.DataFrame(rows, columns=[f"MRR@{K}",f"Recall@{K}",f"NDCG@{K}"]).mean())
    ''',['bm25'],note='Relevansi biner; graded relevance perlu gain 2**rel-1. Semua posisi dipotong K, ID ranking harus unik. Judgment tidak lengkap membuat nilai bias.')
    add('retrieval_dense','E5 dense retrieval: batch per query','model','''
        doc_vectors = embed(docs, role="passage")
        query_vectors = embed(q.tolist(), role="query")
        ranked = [np.argsort(-(doc_vectors @ row))[:10] for row in query_vectors]
    ''',['retrieval_split','embed'],mode='ganti',replaces='bm25',pre=True,note='Dokumen panjang: pecah menjadi chunks dan gabungkan skor max per doc_id; jangan menganggap truncation awal mewakili seluruh dokumen. Vektor sudah L2-normalized.')
    add('weak','Aturan lemah dengan abstain saat kosong/konflik','model','''
        rules = {"dukungan": r"\\b(?:dukung|bermanfaat)\\b", "keluhan": r"\\b(?:keracunan|terlambat)\\b"}
        hits = pd.DataFrame({k:s.str.contains(p, case=False, regex=True) for k,p in rules.items()})
        pseudo = hits.idxmax(axis=1).where(hits.sum(axis=1)==1)  # no hit / konflik = unknown
        print("Coverage (%):", pseudo.notna().mean()*100)
        print("Konflik (%):", hits.sum(axis=1).gt(1).mean()*100)
        for label in rules:
            chosen = df[pseudo.eq(label)]
            print(label, chosen.sample(min(20,len(chosen)), random_state=SEED)[TEXT].tolist())
    ''',['unsup_clean'],note='Aturan ini contoh label isu, bukan otomatis sentimen/stance. Audit gold independen yang tidak dipakai menulis aturan; aturan tidak menjadi ground truth karena coverage tinggi.')
    add('pseudo','Label sedikit: tambah pseudo hanya dari train pool','model','''
        # pool adalah DataFrame UNLABELED yang terpisah dari gold validation.
        u = pool[TEXT].fillna("").astype(str)
        prob = m.predict_proba(v.transform(u))
        chosen = prob.max(1)>=.95
        pseudo_y = m.classes_[prob[chosen].argmax(1)]
        from scipy.sparse import vstack
        xx = vstack([x, v.transform(u[chosen])])
        yy = np.r_[y[tr], pseudo_y]
        from sklearn.base import clone
        candidate = clone(m).fit(xx, yy, sample_weight=np.r_[np.ones(len(tr)), np.full(chosen.sum(),.25)])
        print("Tambahan:", chosen.sum(), "per kelas:", pd.Series(pseudo_y).value_counts())
    ''',['lr'],mode='tambahan',note='Evaluasi candidate pada gold va yang sama. Jangan masukkan gold va ke pool. Duplikat/ID penulis lintas gold-pool harus diperiksa. Threshold confidence bukan bukti label benar; prioritas label umum dapat membesar.')
    add('lsa','LSA: reduksi sparse tanpa densify besar','feat','''
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        dim = min(100, min(x.shape)-1)
        assert dim>=2, "Data/vocabulary terlalu kecil untuk LSA"
        reducer = TruncatedSVD(n_components=dim, random_state=SEED).fit(x)
        z, zv = normalize(reducer.transform(x)), normalize(reducer.transform(xv))
    ''',['tfidf'],mode='tambahan',note='Jangan .toarray() pada matriks TF-IDF besar. SVD fit train saja; representasi test/valid transform dengan reducer yang sama.')
    add('anomaly','Isolation Forest: ranking kandidat anomali','model','''
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(n_estimators=100, contamination="auto", random_state=SEED).fit(z)
        all_z = normalize(reducer.transform(v.transform(s)))
        anomaly = -detector.score_samples(all_z)
        audit = df[[TEXT]].assign(anomaly=anomaly)
        print(audit.nlargest(20,"anomaly").to_string(index=False))
    ''',['lsa'],note='Makin besar skor makin anomali. Tanpa gold, tidak ada F1/accuracy. Anomali topik/panjang bukan otomatis spam atau kesalahan yang boleh dihapus.')
    add('neighbors','Dokumen mirip tanpa label pasangan','model','''
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min(6,len(z)), metric="cosine").fit(z)
        distance, index = neighbors.kneighbors(zv)
        for i in range(min(5,len(va))):
            print("QUERY", s.iloc[va[i]])
            print([(float(d), s.iloc[tr[j]]) for d,j in zip(distance[i], index[i])])
    ''',['lsa'],note='Reference=train, queries=valid: tidak self-match. Ini eksplorasi kemiripan, bukan prediksi skor STS atau bukti duplicate. Jangan membuat semua pasangan n×n.')
    add('keywords','TF-IDF top terms per dokumen','model','''
        matrix = v.transform(s)
        terms = np.array(v.get_feature_names_out())
        keywords = []
        for row in matrix:
            order = row.data.argsort()[-5:][::-1]
            keywords.append(terms[row.indices[order]].tolist())
        print(pd.DataFrame({TEXT:s, "keywords":keywords}).head(10).to_string(index=False))
    ''',['tfidf'],note='Tidak mendensify seluruh vocabulary. Tanpa gold, baca relevansi dan redundansi. Keyword extraction berbeda dari token BIO keyphrase extraction.')
    add('summary','Ringkasan ekstraktif: kalimat paling mirip dokumen','model','''
        from sklearn.feature_extraction.text import TfidfVectorizer
        def summarize(text, n=3):
            sentences = [t.strip() for t in re.split(r"(?<=[.!?])\\s+", str(text)) if t.strip()]
            if len(sentences)<=n: return " ".join(sentences)
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
            matrix = vec.fit_transform(sentences)
            score = np.asarray(matrix @ matrix.mean(axis=0).T).ravel()
            chosen = sorted(score.argsort()[-n:])
            return " ".join(sentences[i] for i in chosen)
        prediction = s.map(summarize)
        print(prediction.head().to_string(index=False))
    ''',['unsup_clean'],note='Urutan kalimat asli dipertahankan. Dokumen sangat panjang: matriks ini n_sentence×vocab, bukan n_document². Batas panjang/coverage jawaban harus mengikuti soal.')
    add('extract','Ekstraksi terstruktur regex + audit','model','''
        records = pd.DataFrame({"url": s.str.findall(r"https?://\\S+"),
            "uang": s.str.findall(r"(?i)rp\\.?\\s*[\\d.,]+(?:\\s*(?:ribu|juta|miliar|triliun))?"),
            "tanggal": s.str.findall(r"\\b\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4}\\b")})
        print(records.head(10).to_string(index=False))
        records.to_json("ekstraksi.jsonl", orient="records", lines=True, force_ascii=False)
    ''',['unsup_clean'],note='Simpan potongan teks bukti. Sebutan lokasi/uang dalam komentar bukan otomatis lokasi kejadian/fakta. Tambahkan kamus domain hanya bila boleh, lalu audit precision per aturan.')
    add('normalize_task','Normalisasi teks: baseline kamus berbatas kata','model','''
        slang = {"yg":"yang", "utk":"untuk", "gak":"tidak"}
        def normalize(text):
            return re.sub(r"\\b\\w+\\b", lambda m: slang.get(m[0],m[0]), text.lower())
        prediction = s.map(normalize)
        print(pd.DataFrame({"raw":s,"normal":prediction}).head(10))
    ''',['unsup_clean'],note='Ini task transformasi output, bukan preprocessing wajib semua model. Pertahankan kapitalisasi/entitas bila evaluator memintanya; kamus dipelajari hanya pada train.')
    add('generation','Seq2seq offline: load model yang sesuai task','model','''
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        MODEL = "C:/gemastik/models/indot5-base"
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        net = AutoModelForSeq2SeqLM.from_pretrained(MODEL, local_files_only=True).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(device)
        @torch.inference_mode()
        def generate(texts):
            output=[]
            for i in range(0,len(texts),4):
                batch=tok(list(texts)[i:i+4],padding=True,truncation=True,max_length=384,return_tensors="pt").to(device)
                output += tok.batch_decode(net.generate(**batch,max_new_tokens=96,num_beams=2),skip_special_tokens=True)
            return output
        prediction = generate(s.tolist())
    ''',['unsup_clean'],pre=True,note='Ganti path dengan model yang SUDAH fine-tuned untuk bahasa/task benar; base T5 bukan penerjemah/ringkasan siap pakai. Prefix/bahasa target mengikuti model. Contoh bukan janji kualitas tanpa fine-tuning.')
    add('gen_metrics','CER/WER untuk teks keluaran','eval','''
        from rapidfuzz.distance import Levenshtein
        refs = df[LABEL].fillna("").astype(str).tolist()
        assert len(prediction)==len(refs)
        cer = sum(Levenshtein.distance(str(p),r) for p,r in zip(prediction,refs))/max(sum(map(len,refs)),1)
        wer = sum(Levenshtein.distance(str(p).split(),r.split()) for p,r in zip(prediction,refs))/max(sum(len(r.split()) for r in refs),1)
        print({"CER":cer,"WER":wer})
    ''',['normalize_task'],note='CER/WER boleh >1. Untuk translation gunakan BLEU/chrF evaluator yang sama; untuk summary gunakan ROUGE evaluator soal. Jangan menamai BLEU buatan sendiri sebagai skor resmi.')
    add('ner_load','NER/POS: tokens dan tags dalam JSON array','load','''
        import json
        import numpy as np, pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        SEED = 42
        df = pd.read_csv("train.csv")
        tokens = df["tokens"].map(json.loads).tolist()
        tags = df["tags"].map(json.loads).tolist()
        assert all(len(a)==len(b) for a,b in zip(tokens,tags))
        groups = df["doc_id"].to_numpy() if "doc_id" in df else np.array([" ".join(s) for s in tokens])
        tr,va = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(tokens,groups=groups))
    ''',note='Jangan lowercase/normalisasi lalu memakai offset lama. Split dokumen/sumber sebelum memecah menjadi kalimat. Jika data CoNLL, parser terpisah harus menjaga batas kalimat.')
    add('ner_crf','CRF fitur token yang singkat','model','''
        import sklearn_crfsuite
        def features(words):
            return [{"w":w.lower(), "title":w.istitle(), "digit":w.isdigit(), "suffix":w[-3:],
                "prev":words[i-1].lower() if i else "<S>",
                "next":words[i+1].lower() if i+1<len(words) else "</S>"} for i,w in enumerate(words)]
        model = sklearn_crfsuite.CRF(max_iterations=80, all_possible_transitions=True)
        model.fit([features(tokens[i]) for i in tr], [tags[i] for i in tr])
        prediction = model.predict([features(tokens[i]) for i in va])
    ''',['ner_load'],note='Memerlukan sklearn-crfsuite yang sudah dipasang offline. POS dinilai per token, NER/ATE/keyphrase BIO dinilai exact entity span, bukan accuracy mayoritas O.')
    add('ner_spans','BIO ke exact span tanpa seqeval','eval','''
        def spans(tags):
            out, start, kind = set(), None, None
            for i,tag in enumerate(list(tags)+["O"]):
                assert tag=="O" or tag.startswith(("B-","I-")), "Perlu skema BIO"
                if start is not None and (tag=="O" or tag.startswith("B-") or tag[2:]!=kind):
                    out.add((kind,start,i)); start=None
                if tag.startswith("B-") or (tag.startswith("I-") and start is None):
                    start,kind=i,tag[2:]
            return out
    ''',['ner_crf'],note='I-X tanpa B-X diperlakukan sebagai awal span; pastikan kebijakan sesuai evaluator. BILOU memerlukan decoder berbeda. Tidak memerlukan seqeval yang tidak tercatat di bundle.')
    add('ner_eval','Entity exact-span F1 dengan support','eval','''
        true_positive, n_pred, n_gold = 0,0,0
        for i,guess in zip(va,prediction):
            gold, pred = spans(tags[i]), spans(guess)
            true_positive += len(gold & pred)
            n_pred += len(pred); n_gold += len(gold)
        precision = true_positive/max(n_pred,1); recall=true_positive/max(n_gold,1)
        print({"precision":precision, "recall":recall, "F1":2*true_positive/max(n_pred+n_gold,1),
            "gold_entities":n_gold, "pred_entities":n_pred})
    ''',['ner_spans'],note='Exact batas dan tipe wajib sama. POS memakai pos_eval, bukan span. Support entitas langka/BIO ilegal perlu dilihat; truncation tidak boleh diisi O diam-diam.')
    add('qa_load','QA: pertanyaan, konteks, jawaban','load','''
        import re, json
        import numpy as np, pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        SEED = 42
        df = pd.read_csv("train.csv")
        Q,C,ANSWER = "question","context","answer"
        assert df[[Q,C]].notna().all().all()
        groups = pd.factorize(df[C])[0]
        tr,va = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(df,groups=groups))
    ''',note='Konteks sama tidak lintas split. Jika ada answer_start, jangan mengubah konteks/encoding setelah offset ditetapkan. Tentukan apakah ada soal tidak terjawab.')
    add('qa_baseline','QA tanpa pretrained: kalimat kandidat','model','''
        from sklearn.feature_extraction.text import TfidfVectorizer
        def answer(question, context):
            sentences = [t for t in re.split(r"(?<=[.!?])\\s+", context) if t.strip()]
            if not sentences: return ""
            vec = TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4))
            x = vec.fit_transform(sentences)
            score = (vec.transform([question]) @ x.T).toarray().ravel()
            return sentences[score.argmax()]
        prediction = [answer(df.iloc[i][Q],df.iloc[i][C]) for i in va]
    ''',['qa_load'],note='Ini baseline sentence retrieval, bukan jawaban span presisi. F1 jawaban pendek akan rendah. Tidak menggunakan jawaban gold untuk memilih kalimat.')
    add('qa_reader','QA pretrained lokal dengan stride','model','''
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
        MODEL = "C:/gemastik/models/qa-indobert"
        tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        net = AutoModelForQuestionAnswering.from_pretrained(MODEL,local_files_only=True)
        reader = pipeline("question-answering", model=net, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
        batch = [{"question":df.iloc[i][Q], "context":df.iloc[i][C]} for i in va]
        output = reader(batch, batch_size=8, doc_stride=128, max_seq_len=384, max_answer_len=40)
        prediction = [row["answer"] for row in output]
    ''',['qa_load'],mode='ganti',replaces='qa_baseline',pre=True,note='Model harus sudah punya head QA terlatih. SQuAD2/no-answer memerlukan handling/threshold null yang divalidasi. Jangan argmax start/end terpisah tanpa span constraints.')
    add('qa_eval','QA: exact match dan token F1 multi-jawaban','eval','''
        from collections import Counter
        def words(text): return re.findall(r"\\w+", str(text).lower())
        def f1_answer(pred, ref):
            p,r = words(pred),words(ref)
            overlap = sum((Counter(p)&Counter(r)).values())
            return 2*overlap/(len(p)+len(r)) if p or r else 1.
        scores=[]
        for i,pred in zip(va,prediction):
            refs=df.iloc[i][ANSWER]; refs=refs if isinstance(refs,list) else [refs]
            scores.append([max(words(pred)==words(r) for r in refs), max(f1_answer(pred,r) for r in refs)])
        print(pd.DataFrame(scores,columns=["EM","token_F1"]).mean())
    ''',['qa_baseline'],note='Jika jawaban list tersimpan sebagai JSON string, decode dahulu. Normalisasi harus mengikuti evaluator. Jangan mengganti jawaban missing menjadi literal "nan".')
    add('nextword','Next word: baseline bigram train saja','model','''
        from collections import Counter, defaultdict
        table = defaultdict(Counter); unigram=Counter()
        for text in s.iloc[tr]:
            row = text.lower().split(); unigram.update(row)
            for a,b in zip(row,row[1:]): table[a][b]+=1
        def complete(text, k=5):
            row = text.lower().split()
            source = table.get(row[-1],unigram) if row else unigram
            return [word for word,_ in source.most_common(k)]
        print([complete(text) for text in s.iloc[va].head(5)])
    ''',['unsup_clean','split'],note='Untuk mengukur Top-K, input valid harus prefix dan labelnya kata berikutnya; jangan memberi kalimat lengkap termasuk jawaban. Pisahkan dokumen sebelum membuat prefix.')
    add('plot_bar','Grafik sederhana bila dibutuhkan slide','out','''
        import matplotlib.pyplot as plt
        values = df[LABEL].value_counts()
        values.plot.bar(color="#0085ff", figsize=(7,3))
        plt.ylabel("Jumlah"); plt.tight_layout()
        plt.savefig("kelas.png",dpi=180,bbox_inches="tight"); plt.show()
    ''',['load'],mode='tambahan',note='Angka tetap tersedia di EDA; grafik hanya saat membantu presentasi. Palet tambahan #fcd12a. Tidak perlu donut/card dekoratif.')
    add('save','Simpan model/vectorizer dan hasil secara sederhana','out','''
        import joblib
        joblib.dump({"vectorizer":v, "model":m, "text_col":TEXT}, "baseline.joblib")
        pd.DataFrame({"row":va, "true":y[va], "pred":pred}).to_csv("validation.csv",index=False)
    ''',['lr'],mode='tambahan',note='Reload hanya file tepercaya; versi paket saat reload harus cocok. Simpan label order/threshold jika ada. Untuk head embedding, encoder path juga wajib dicatat.')
    add('pos_eval','POS: metrik per token, bukan BIO span','eval','''
        from sklearn.metrics import classification_report
        gold = [tag for i in va for tag in tags[i]]
        guess = [tag for row in prediction for tag in row]
        assert len(gold)==len(guess)
        print(classification_report(gold,guess,zero_division=0))
    ''',['ner_crf'],note='POS tidak menggunakan B-/I- prefix. Laporkan kelas jarang serta accuracy; jangan memanggil evaluator entity span pada POS.')
    add('identity','Transformasi teks: baseline salin input','model','''
        prediction = s.tolist()
        print(prediction[:5])
    ''',['unsup_clean'],note='Pembanding minimum untuk normalisasi/translation. Salin input bukan penerjemah. Jika tidak ada model/dataset bahasa target, tidak ada solusi terjemahan umum yang valid hanya dengan regex.')
    add('rank_load','Ranking berlabel: qid, query, document, relevance','load','''
        import numpy as np, pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        SEED=42
        df=pd.read_csv("train.csv")
        a,b=df["query"].fillna("").astype(str),df["document"].fillna("").astype(str)
        y=pd.to_numeric(df["relevance"],errors="raise").to_numpy()
        assert (y>=0).all() and np.equal(y,y.astype(int)).all()
        groups=df["qid"].to_numpy()
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(df,groups=groups))
    ''',note='Label graded relevance nonnegatif, bukan class name. Seluruh kandidat satu query harus bersama. Sediakan semua dokumen relevan bila akan melaporkan recall.')
    add('rank_model','LGBMRanker dengan fitur pasangan','model','''
        import lightgbm as lgb
        order=np.argsort(groups[tr],kind="stable")
        _,sizes=np.unique(groups[tr][order],return_counts=True)
        m=lgb.LGBMRanker(objective="lambdarank",n_estimators=200,verbosity=-1)
        m.fit(x[order],y[tr][order],group=sizes)
        score=m.predict(xv)
        ranking=df.iloc[va][["qid","relevance"]].assign(score=score).sort_values(["qid","score"],ascending=[True,False])
        print(ranking.head(20).to_string(index=False))
    ''',['rank_load','pair_features'],note='Ukuran group sesuai baris X yang diurutkan. Nilai model adalah skor ranking, bukan probabilitas. Fitur sparse tidak diubah menjadi dense seluruh vocabulary.')
    add('rank_eval','NDCG graded per query','eval','''
        from sklearn.metrics import ndcg_score
        values=[]
        for _,part in ranking.groupby("qid"):
            if len(part)<2 or not part.relevance.gt(0).any(): continue
            gains=2.**part.relevance.to_numpy()-1
            values.append(ndcg_score([gains],[part.score.to_numpy()],k=10))
        print("NDCG@10:",np.mean(values),"query dievaluasi:",len(values))
    ''',['rank_model'],note='sklearn ndcg_score memakai gain yang diberikan; bentuk 2**rel-1 mengikuti definisi soal. Query zero-gold/1 kandidat dilaporkan tidak dinilai, bukan disembunyikan sebagai accuracy tinggi.')
    add('pair_eda','Pair: panjang kedua sisi dan shortcut lexical','eda','''
        left,right = a.str.split().str.len(),b.str.split().str.len()
        print(pd.DataFrame({"left":left,"right":right,"abs_diff":abs(left-right)}).describe().T)
        overlap = [len(set(u.lower().split()) & set(w.lower().split()))/max(len(set(u.lower().split()) | set(w.lower().split())),1) for u,w in zip(a,b)]
        print(pd.Series(overlap).describe())
        print("Duplicate ordered pair:", pd.DataFrame({"a":a,"b":b}).duplicated().sum())
    ''',['pair_clean'],note='Kemiripan tinggi bukan entailment; angka/negasi dapat mengubah relasi. Bila hypothesis-only sudah tinggi, curigai artefak anotasi. Teks kosong harus ditangani sebelum connected grouping.')
    add('ret_eda','Retrieval: coverage, panjang, duplicate, relevance','eda','''
        print(corpus.isna().sum()); print(queries.isna().sum())
        print("Duplicate text:", pd.Series(docs).str.lower().duplicated().sum())
        print("Panjang dokumen:", pd.Series(docs).str.split().str.len().describe(percentiles=[.5,.9,.99]))
        print("Relevan per query:", pd.Series(list(map(len,gold))).describe())
        print("Query duplicate:", queries["query"].fillna("").str.lower().duplicated().sum())
    ''',['retrieval_load'],note='Gold harus ada dalam corpus. Banyak gold per query bukan error duplicate. Ukur apakah jawaban relevan berada setelah truncation bila memakai encoder/reranker.')
    add('token_eda','Token tagging: panjang, kelas, missing/alignment','eda','''
        from collections import Counter
        print("Kalimat / token:", len(tokens), sum(map(len,tokens)))
        print(pd.Series(list(map(len,tokens))).describe(percentiles=[.5,.95,.99]))
        print(Counter(tag for row in tags for tag in row))
        print("Sentence duplicate:", pd.Series([" ".join(row) for row in tokens]).duplicated().sum())
        assert all(all(isinstance(w,str) for w in row) for row in tokens)
    ''',['ner_load'],note='Jangan menyamakan jumlah tag O dengan kualitas NER. Audit rare entity, BIO ilegal, token terpotong, dan dokumen lintas split.')
    add('qa_eda','QA: coverage jawaban dan panjang konteks','eda','''
        print(df[[Q,C,ANSWER]].isna().sum())
        print(df[C].str.split().str.len().describe(percentiles=[.5,.95,.99]))
        print("Konteks unik:", df[C].nunique(), "pertanyaan duplicate:", df[Q].duplicated().sum())
        if "answer_start" in df:
            known=df["answer_start"].notna() & df[ANSWER].notna()
            match=[str(row[C])[int(row.answer_start):int(row.answer_start)+len(str(row[ANSWER]))]==str(row[ANSWER]) for _,row in df[known].iterrows()]
            print("Offset jawaban cocok:", np.mean(match), "n:",len(match))
    ''',['qa_load'],note='Untuk format answers JSON list/dict, decode dahulu. No-answer tidak sama dengan annotation missing. Jika offset mismatch, perbaiki data/preprocessing dahulu, bukan memaksa span.')


def flows():
    common=['load','profile','length','vocab','noise']
    supervised=common+['target','clean','split','tfidf','lr','evaluate']
    unsup=common+['unsup_clean','split']
    classification=['formats','normalize','user_groups','time_split','char','union','svm','dummy','word2vec',
        'embed','embed_head','tweet_prep','bert_setup','token_length','bert_train','bert_predict',
        'threshold','class_bias','ensemble','calibrate','final','final_post','language','plot_bar','save']
    result=[]
    def add(key,title,tasks,priority,core,extra,note):
        result.append({'id':key,'title':title,'tasks':tasks.split(),'priority':priority,'core':core,'extra':extra,'note':note})
    add('classification','Sentimen / stance / klasifikasi','sentiment stance binary multiclass spam emotion fakenews intent',1,
        supervised,classification,'Paling dahulu disiapkan: satu label per teks. Sentimen tidak sama dengan stance; teks negatif tentang pelaksanaan belum tentu menolak program.')
    add('multilabel','Multilabel / banyak tag','multilabel',2,common+['ml_clean','ml_eda','split','tfidf','ml_model','ml_eval'],
        ['ml_list','user_groups','time_split','char','union','ml_threshold','ml_final','language'],
        'Beberapa label dapat aktif bersamaan; anotasi unknown harus berbeda dari negatif. Jangan memakai workflow satu-label atau argmax.')
    add('absa','Sentimen per aspek (ABSA)','absa',3,common+['absa_clean','split','tfidf','absa_model'],
        ['user_groups','time_split','char','union','language'], 'Kolom aspek bisa kategorikal, biner, atau list pasangan. Contoh inti memakai sentimen kategorikal per aspek; pilih multilabel jika setiap aspek hanya 0/1.')
    add('topic','Topik dari teks tanpa label','topic',4,unsup+['topic_model','topic_eval'],
        ['lda','language','keywords'], 'Cari tema yang bisa dijelaskan dan dibuktikan contoh teks. Pilih K pada bagian development; nama topik bukan label kebenaran.')
    add('clustering','Clustering teks','doc_cluster emb_cluster',5,unsup+['tfidf','cluster','cluster_eval'],
        ['embed','language'], 'Bandingkan bentuk cluster, contoh, support dan stabilitas. Jangan memaksakan semua outlier menjadi cluster normal.')
    add('temporal','Tren topik / sentimen dan laporan','trend report',6,common+['unsup_clean','time_split','topic_model','topic_eval','temporal'],
        ['weak','extract','language'], 'Gunakan waktu lama untuk fit, waktu baru untuk audit. Model topik tetap; bandingkan proporsi dan jumlah, jangan menyimpulkan sebab-akibat dari timeline.')
    add('weak','Tanpa gold / weak supervision','weak',7,common+['unsup_clean','weak'],
        ['language','extract','topic_model'], 'Ukuran sukses awal adalah coverage, konflik dan audit precision pada gold independen. Skor melawan pseudo-label sendiri bukan generalisasi.')
    add('semi','Gold sedikit / pseudo-label / self-training','pseudo selftrain confidence iterative',8,supervised,
        ['pseudo','user_groups','time_split','embed','embed_head'], 'Selesaikan baseline gold dahulu. Pool unlabeled terpisah dari gold validation, duplicate/penulis diperiksa. Tambahan data wajib diuji terhadap baseline yang sama.')
    add('pair','NLI / pair matching / parafrase','pair consistency dupq',9,
        ['load','pair_clean','pair_eda','length','vocab','noise','pair_split','pair_features','lr','evaluate'],['class_bias','language','embed'],
        'NLI tiga kelas berbeda dari duplicate biner. Gunakan label data sesuai soal; simetri hanya untuk hubungan yang simetris. Fitur kedua sisi tetap terpisah.')
    add('similarity','Semantic similarity berskor kontinu','semsim',10,
        ['load','sts_clean','pair_eda','length','vocab','noise','pair_split','pair_features','sts_model'],['embed','language'],
        'Skor kemiripan kontinu: Pearson/Spearman dan kesalahan skala. Jangan menggunakan class label atau ID pasangan sebagai target.')
    add('retrieval','Retrieval / pencarian dokumen','ir',11,
        ['retrieval_load','ret_eda','retrieval_split','bm25','retrieval_eval'],['embed','retrieval_dense'],
        'Jawaban adalah ranking doc_id, bisa banyak dokumen relevan. Group query, cutoff K dan definisi relevance mengikuti soal.')
    add('ranking','Ranking dengan relevance berjenjang','ranking',12,
        ['rank_load','pair_features','rank_model','rank_eval'],[],
        'Skema query-doc berlabel graded. Semua kandidat query bersama di split, fitur fit train saja, group size ranker sesuai urutan baris.')
    add('regression','Regresi skor / engagement / ordinal','regression',13,
        common+['reg_clean','split','tfidf','ridge'],['time_split','user_groups','char','union','word2vec'],
        'Baca metrik soal: RMSE, MAE, RMSLE dan QWK membutuhkan keputusan berbeda. Jangan menganggap semua target float adalah regresi.')
    add('neighbors','Representasi / kemiripan dokumen / reduksi','doc_sim repr dimred',14,
        unsup+['tfidf','lsa','neighbors'],['embed','language'],
        'Tanpa skor gold pasangan, nearest-neighbor hanya eksplorasi. Hindari self-match dan matriks jarak n×n; simpan mapping baris asli.')
    add('anomaly','Anomali / spam tak berlabel','anomaly',15,
        unsup+['tfidf','lsa','anomaly'],['language','weak'],
        'Model mengurutkan kandidat tidak biasa; bukan alasan menghapusnya otomatis. Baca contoh per panjang/bahasa/sumber.')
    add('keyword','Kata kunci tanpa label','keyword',16,
        unsup+['tfidf','keywords'],['language'],
        'Keyword bebas berbeda dari token tagging. Evaluasi perlu gold keyphrase; tanpa gold laporkan audit relevansi/duplikasi.')
    add('qa','Question answering / span','qa',17,
        ['qa_load','qa_eda','qa_baseline','qa_eval'],['qa_reader'],
        'Tetap siapkan walau di luar klasifikasi: jenis final dapat berubah. Jaga konteks dan offset, jawaban ganda, no-answer dan stride dokumen panjang.')
    add('ner','NER / term aspek / keyphrase BIO','ner ate keyphrase',18,
        ['ner_load','token_eda','ner_crf','ner_spans','ner_eval'],[],
        'Batas kalimat/dokumen, alignment token-tag dan skema BIO adalah inti. Offset tidak boleh berubah karena normalisasi agresif.')
    add('pos','POS tagging','pos',19,['ner_load','token_eda','ner_crf','pos_eval'],[],
        'POS dinilai per token; jangan memanggil evaluator entity span BIO untuk kelas kata.')
    add('summary','Ringkasan ekstraktif / abstraktif','summarization',20,
        common+['unsup_clean','summary'],['generation'], 'Mulai ekstraktif. Model abstraktif harus sesuai task, dinilai dengan rujukan dan audit fakta/negasi; ROUGE bukan jaminan tidak berhalusinasi.')
    add('extract','Ekstraksi informasi dan bukti domain','mbg_extract',21,
        common+['unsup_clean','extract'],['weak','language'], 'Ekstrak apa yang disebut teks beserta bukti. Jangan menyamakan komentar dengan catatan kejadian terverifikasi.')
    add('normalization','Normalisasi sebagai task output','normalization',22,
        common+['unsup_clean','normalize_task'],['gen_metrics','generation'], 'Kamus dan model dipelajari dari train. Jangan mengganti ejaan yang sebenarnya nama/tempat/istilah daerah tanpa bukti.')
    add('translation','Terjemahan','translation',23,
        common+['unsup_clean','identity'],['generation','gen_metrics'], 'Baseline salin hanya pembanding. Memerlukan data paralel/model bahasa target yang sesuai; tidak ada penerjemah umum dari regex.')
    add('nextword','Prediksi kata berikutnya','nextword',24,
        unsup+['nextword'],[], 'Split dokumen sebelum membuat contoh prefix. Pengujian tidak boleh menyertakan kata jawaban di input.')
    add('language','Identifikasi bahasa','langdetect',25,
        common+['language'],['clean','split','tfidf','lr','evaluate'], 'Tanpa gold, laporkan estimasi, ketidakpastian dan audit campuran bahasa. Jika ada label bahasa gold, gunakan pipeline supervised tambahan.')
    return result
