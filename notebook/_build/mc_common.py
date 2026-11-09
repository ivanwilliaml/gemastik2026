"""Bagian bersama tiga notebook multiclass: kepala (data → EDA) dan ekor (evaluasi).

Generator ini hanya merakit sel notebook. Kode di dalam notebook sengaja datar:
tanpa kelas, fungsi hanya bila dipakai berulang, dan satu baris bila cukup satu baris.
"""
from pathlib import Path
from textwrap import dedent
import ast, base64, json, tempfile
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "b_klasifikasi/10_multi_kelas"
STEPWISE = (ROOT / "preprocessing_universal/stepwise.py").read_text(encoding="utf-8")
RESOURCES = STEPWISE.split("# %% resources\n", 1)[1].split("\n\ndef map_words", 1)[0]
DATASET = "dzikriraihan/indonesian-youtube-comments-mbg-program"


class Book:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nb.v4.new_markdown_cell(dedent(s).strip()))

    def code(self, s):
        self.cells.append(nb.v4.new_code_cell(dedent(s).strip()))

    def config(self, note, s):
        self.md("**Konfigurasi** · " + note)
        self.code(s)


def head(book, title, intro, gpu, stem=False):
    book.md(f"# {title}\n\n{intro}")
    book.md("## 1. Konfigurasi dan lingkungan")
    book.config("Satu tempat untuk semua pengaturan umum. `N_ROWS=None` memakai seluruh CSV. `GROUP_COL` boleh `None`; isi dengan kolom yang tidak boleh bocor antar split (pengguna, video, thread).", r'''
        SEED, N_ROWS, TOP = 42, 5000, 15
        DATA_FILE = None                                  # None: cari comment_sentiment.csv di /kaggle/input
        TEXT_COL, LABEL_COL = "Comment", "Sentiment"
        CONF_COL, GROUP_COL = "Confidence Score", "Commenter ID"
        NEAR_DUP, NEAR_THRESHOLD = True, 0.9              # grup near-duplicate: MinHash char 5-gram
        CLASS_WEIGHT, MIN_POST_GAIN = "balanced", 0.005   # bobot kelas; syarat minimal kenaikan post-processing
        OUT = "/kaggle/working/hasil"
    ''')
    book.md("Paket tambahan dipasang di Kaggle (internet aktif). Semua komputasi berjalan di Kaggle, bukan di laptop.")
    book.code(r'''
        import sys, subprocess, os, warnings
        assert os.path.exists("/kaggle/input"), "Jalankan notebook ini di Kaggle."
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gensim>=4.4,<5", "ftfy>=6.3,<7", "emoji>=2.14,<3",
                        "datasketch>=1.6,<3", "langid==1.1.6", "Sastrawi"], check=True)
        os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1", USE_TF="0", USE_FLAX="0")
        warnings.filterwarnings("ignore")
    ''')
    book.code(r'''
        import re, html, json, time, random, hashlib, unicodedata
        from pathlib import Path
        from collections import Counter
        import numpy as np, pandas as pd, torch, emoji, ftfy
        from IPython.display import display
        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import StratifiedGroupKFold

        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); Path(OUT).mkdir(parents=True, exist_ok=True)
        pd.set_option("display.max_columns", None, "display.max_colwidth", 120, "display.width", 200)
        show = lambda title, t: (print("\n" + title), display((t.to_frame() if isinstance(t, pd.Series) else t).round(3)))
        dist = lambda v: pd.DataFrame({"n": v.value_counts(dropna=False), "%": v.value_counts(dropna=False, normalize=True) * 100})
        START = time.perf_counter()
    ''' + ('''
        assert torch.cuda.is_available(), "Aktifkan GPU Kaggle (Accelerator → GPU T4)."
        print("GPU:", torch.cuda.get_device_name(0))
    ''' if gpu else ""))

    # ------------------------------------------------------------------ data
    book.md("## 2. Muat data dan audit kualitas sebelum membuang apa pun\n\nAudit dilakukan **sebelum** baris invalid dibuang, supaya jumlah yang hilang terlihat. Kolom tidak ditebak: kalau nama kolom salah, `assert` memberi daftar kolom yang ada.")
    book.code(r'''
        path = Path(DATA_FILE) if DATA_FILE else next(Path("/kaggle/input").rglob("comment_sentiment.csv"))
        raw = pd.read_csv(path)
        assert {TEXT_COL, LABEL_COL} <= set(raw), f"Kolom tersedia: {raw.columns.tolist()}"
        show("Skema sebelum pembersihan", pd.DataFrame({"tipe": raw.dtypes.astype(str), "missing": raw.isna().sum(),
                                                         "missing_%": raw.isna().mean() * 100, "unik": raw.nunique()}))
        print(path, "| baris:", len(raw), "| duplikat baris penuh:", raw.duplicated().sum())
    ''')
    book.code(r'''
        is_str = raw[TEXT_COL].map(lambda x: isinstance(x, str))
        txt = raw[TEXT_COL].where(is_str, "")
        bad = pd.DataFrame({"teks_null": raw[TEXT_COL].isna(), "teks_non_string": raw[TEXT_COL].notna() & ~is_str,
                            "teks_kosong": txt.str.strip().eq(""), "teks_pseudo_null": txt.str.strip().str.casefold().isin(["nan", "null", "none", "n/a", "-"]),
                            "label_null": raw[LABEL_COL].isna(), "label_kosong": raw[LABEL_COL].astype("string").str.strip().eq("").fillna(False)})
        show("Kualitas input · flag boleh overlap", pd.DataFrame({"n": bad.sum(), "%": bad.mean() * 100}))
        data = raw.loc[~bad.any(axis=1)].reset_index(names="row_id")
        data[LABEL_COL] = data[LABEL_COL].astype(str).str.strip()
        if N_ROWS: data = data.sample(min(N_ROWS, len(data)), random_state=SEED).reset_index(drop=True)
        print("dibuang karena invalid:", int(bad.any(axis=1).sum()), "| baris dipakai:", len(data))
    ''')

    # --------------------------------------------------------- preprocessing
    book.md("""## 3. Preprocessing universal, langkah demi langkah

Sumbernya `preprocessing_universal/stepwise.py`, dipecah jadi daftar langkah satu baris. Rantainya satu arah: `teks asli → safe → normalized → clean`. Setiap langkah dicatat berapa baris yang berubah, jadi langkah yang tidak ada gunanya di data ini kelihatan (berubah = 0).

| View | Isi | Dipakai untuk |
|---|---|---|
| `safe` | perbaikan teknis saja: encoding, unicode, entity HTML, karakter tak terlihat, spasi | dasar semua view |
| `minimal` | `safe` + huruf kecil | char n-gram, FastText |
| `normalized` | URL/mention/email/telepon → placeholder, slang, huruf berulang, tawa, emoji → teks | TF-IDF kata, Word2Vec |
| `clean` | `normalized` + tanda baca, kata berulang, stopword, stemming | bag-of-words klasik |
| `embed` | `safe` + placeholder + emoji → teks, **huruf besar dipertahankan** | encoder pretrained |
| `bert` | resep model card IndoBERTweet: huruf kecil, `HTTPURL`, `@USER`, emoji → teks | IndoBERTweet |

Negasi (`tidak`, `bukan`, `belum`, ...) dan istilah domain (`mbg`, `sppg`, ...) dilindungi dari stopword, stemming, dan koreksi.""")
    book.md("Kamus dan pola regex dibaca langsung dari file universal. **Tinjau kamus ini untuk domain baru**: `SLANG`, `PROTECTED`, `VOCAB`.")
    book.code(RESOURCES + "\n" + 'words = lambda text, f: TOKEN.sub(lambda m: f(m[0]) if WORD.fullmatch(m[0]) else m[0], text)   # terapkan f hanya pada kata biasa')
    book.config("`TYPOS` diisi setelah membaca daftar kata langka di EDA (bagian 6): `{\"bgtt\": \"banget\"}`. `CLEAN_OPT` mengatur view `clean`; emoji dan emoticon dipertahankan karena membawa sentimen.", r'''
        TYPOS = {}                                     # koreksi typo hasil review manual
        REGIONAL_ON = False                            # True: pakai kamus bahasa daerah (ora, teu, indak, apik)
        CLEAN_OPT = dict(emoji=False, emoticon=False, angka=False, stopword=True, stem=''' + str(stem) + r''')   # True = HAPUS/aktifkan
        STEM_MIN_DF = 2                                # stemming hanya untuk kata yang muncul >= 2 kali (Sastrawi lambat; kata tunggal tidak berguna di TF-IDF min_df=2)
    ''')
    book.md("Langkah **safe**: hanya memperbaiki yang rusak secara teknis, tidak mengubah makna.")
    book.code(r'''
        SAFE = [
            ("encoding",    lambda s: s.map(lambda t: ftfy.fix_text(t, unescape_html=False, uncurl_quotes=False, fix_latin_ligatures=False,
                                                                       fix_character_width=False, remove_control_chars=False, normalization=None))),
            ("unicode_nfc", lambda s: s.map(lambda t: unicodedata.normalize("NFC", t))),
            ("html_entity", lambda s: s.map(html.unescape)),
            ("tak_terlihat", lambda s: s.str.replace(r"[﻿​⁠\x00]", "", regex=True).str.replace(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", regex=True)),
            ("spasi",       lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip())]
    ''')
    book.md("Langkah **normalized**: menyeragamkan variasi tulisan. Slang, huruf berulang, dan koreksi memakai kamus sehingga kata domain tidak ikut rusak.")
    book.code(r'''
        NORMAL = [
            ("obfuscation", lambda s: s.str.replace(r"(?<!\w)[a-zA-Z][a-zA-Z0-9@*$]{3,}(?!\w)", lambda m: OBFUSCATION.get(m[0].casefold(), m[0]), regex=True)),
            ("url",         lambda s: s.str.replace(URL, lambda m: " URL " + m[0][len(m[0].rstrip(".,!?;)")):], regex=True)),
            ("email",       lambda s: s.str.replace(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " EMAIL ", regex=True)),
            ("telepon",     lambda s: s.str.replace(r"(?<!\w)(?:\+62|62|0)[ -]?8(?:[ -]?\d){8,11}(?!\w)", " PHONE ", regex=True)),
            ("mention",     lambda s: s.str.replace(r"(?<![\w@])@[\w.]+", " USER ", regex=True)),
            ("hashtag",     lambda s: s.str.replace(r"(?<!\w)#(\w+)", lambda m: " " + re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", m[1]) + " ", regex=True)),
            ("casefold",    lambda s: s.str.casefold()),
            ("slang",       lambda s: s.map(lambda t: words(t, lambda w: SLANG.get(w.casefold(), w) if w.casefold() not in PROTECTED | AMBIGUOUS else w))),
            ("huruf_ulang", lambda s: s.map(lambda t: words(t, lambda w: w if w.casefold() in PROTECTED else next(
                                (c for c in (re.sub(r"([^\W\d_])\1{2,}", r, w, flags=re.I) for r in (r"\1", r"\1\1")) if c.casefold() in VOCAB), w)))),
            ("tawa",        lambda s: s.map(lambda t: words(t, lambda w: "wkwk" if re.fullmatch(r"(?:wk){2,}|w+k+w+k+[wk]*", w, re.I)
                                                              else "haha" if re.fullmatch(r"(?:ha){2,}h?", w, re.I) else w))),
            ("reduplikasi", lambda s: s.str.replace(r"\b\w+2\b", lambda m: {"anak2": "anak-anak", "orang2": "orang-orang", "teman2": "teman-teman"}.get(m[0].casefold(), m[0]), regex=True)),
            ("daerah",      lambda s: s.map(lambda t: words(t, lambda w: REGIONAL.get(w.casefold(), w))) if REGIONAL_ON else s),
            ("typo",        lambda s: s.map(lambda t: words(t, lambda w: TYPOS.get(w.casefold(), w) if w.casefold() not in PROTECTED | AMBIGUOUS else w))),
            ("tanda_ulang", lambda s: s.str.replace(r"([!?.,])\1{2,}", r"\1\1", regex=True)),
            ("emoji_ke_teks", lambda s: s.map(lambda t: emoji.demojize(t, delimiters=(" ", " ")))),
            ("spasi",       lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip())]
    ''')
    book.md("Langkah **clean**: untuk model bag-of-words. Stemming hanya pada kata yang tidak dilindungi, dan hasilnya di-cache per kata supaya cepat.")
    book.code(r'''
        from functools import lru_cache
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stemmer = StemmerFactory().create_stemmer()
        stem = lru_cache(None)(lambda w: w if w in PROTECTED | NEGATIONS else stemmer.stem(w))
        STOP_BLOCK = STOPWORDS - NEGATIONS - PROTECTED
        CLEAN = [
            ("hapus_emoji",    lambda s: s.map(lambda t: emoji.replace_emoji(t, " ")) if CLEAN_OPT["emoji"] else s),
            ("hapus_emoticon", lambda s: s.str.replace(r"(?<!\w)(?:[:;=8][\-^']?[)(/DPp]+|<3)(?!\w)", " ", regex=True) if CLEAN_OPT["emoticon"] else s),
            ("hapus_angka",    lambda s: s.str.replace(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)", " ", regex=True) if CLEAN_OPT["angka"] else s),
            ("tanda_baca",     lambda s: s.str.replace(r"[^\w\s]|_", " ", regex=True)),
            ("kata_berulang",  lambda s: s.str.replace(r"\b(\w+)(?:\s+\1\b)+", r"\1", regex=True, flags=re.I)),
            ("stopword",       lambda s: s.map(lambda t: " ".join(w for w in t.split() if w.casefold() not in STOP_BLOCK)) if CLEAN_OPT["stopword"] else s),
            ("stemming",       lambda s: (lambda df: s.map(lambda t: " ".join(stem(w) if df[w] >= STEM_MIN_DF else w for w in t.split())))(Counter(w for t in s for w in set(t.split()))) if CLEAN_OPT["stem"] else s),
            ("spasi",          lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip())]
    ''')
    book.md("Jalankan rantai. Tabel menunjukkan **berapa baris berubah di tiap langkah**: itulah bukti langkah mana yang relevan untuk data ini.")
    book.code(r'''
        log, s = [], data[TEXT_COL]
        for view, steps in [("safe", SAFE), ("normalized", NORMAL), ("clean", CLEAN)]:
            for name, f in steps:
                new = f(s); log.append({"view": view, "langkah": name, "baris_berubah": int((new != s).sum()), "%": (new != s).mean() * 100}); s = new
            data[view] = s
            if view == "safe": data = data.loc[data.safe.ne("")].reset_index(drop=True); s = data.safe
        show("Dampak tiap langkah preprocessing · semua baris", pd.DataFrame(log).set_index(["view", "langkah"]))
    ''')
    book.code(r'''
        data["minimal"] = data.safe.str.casefold()
        data["bert"] = (data.safe.str.lower().str.replace(URL, "HTTPURL", regex=True).str.replace(r"(?<![\w@])@[\w.]+", "@USER", regex=True)
                        .map(emoji.demojize).str.replace(r"\s+", " ", regex=True).str.strip())
        EMBED_STEPS = {"url", "email", "telepon", "mention", "hashtag", "emoji_ke_teks", "spasi"}
        s = data.safe
        for name, f in NORMAL:
            if name in EMBED_STEPS: s = f(s)
        data["embed"] = s
        VIEWS = ["safe", "minimal", "normalized", "clean", "embed", "bert"]
        show("Ringkasan view", pd.DataFrame({v: {"berubah_dari_safe_%": data[v].ne(data.safe).mean() * 100, "kosong": int(data[v].eq("").sum()),
                                                  "unik": data[v].nunique(), "kata_p50": data[v].str.split().str.len().median(),
                                                  "kata_p95": data[v].str.split().str.len().quantile(.95)} for v in VIEWS}).T)
        show("Sebelum → sesudah", data[[TEXT_COL, "normalized", "clean", "embed", "bert"]].sample(5, random_state=SEED))
    ''')

    # ----------------------------------------------------------------- split
    book.md("""## 4. Grup duplikat, lalu split train / validation / holdout

Komentar yang sama (atau hampir sama) tidak boleh berada di dua split berbeda, karena skor jadi bocor. Tiga hal disatukan dalam satu grup: **teks identik**, **pengguna yang sama** (`GROUP_COL`), dan **near-duplicate** (MinHash char 5-gram, Jaccard ≥ `NEAR_THRESHOLD`). Gabungannya dicari lewat connected components.

Split 60 / 20 / 20 dengan `StratifiedGroupKFold`: proporsi kelas terjaga dan tidak ada grup yang terbelah. **Validation** untuk memilih model dan parameter; **holdout** hanya dibuka satu kali di akhir.""")
    book.code(r'''
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        n = len(data); key = data.safe.str.casefold().str.replace(r"\s+", " ", regex=True)
        a = key.factorize()[0]                                                          # grup teks identik
        b = data[GROUP_COL].astype(str).factorize()[0] if GROUP_COL else np.arange(n)   # grup pengguna / thread
        edges = [np.c_[np.arange(n), n + a], np.c_[np.arange(n), n + a.max() + 1 + b]]
        if NEAR_DUP:
            from datasketch import MinHash, MinHashLSH
            lsh, grams, near = MinHashLSH(threshold=NEAR_THRESHOLD, num_perm=64), {}, []
            for i, k in key.drop_duplicates().items():
                if len(k) < 20: continue                                                # teks pendek: hanya exact
                g = {k[j:j + 5] for j in range(len(k) - 4)}; m = MinHash(num_perm=64, seed=SEED); m.update_batch([x.encode() for x in sorted(g)])
                near += [(i, j) for j in lsh.query(m) if len(g & grams[j]) / len(g | grams[j]) >= NEAR_THRESHOLD]
                lsh.insert(i, m); grams[i] = g
            if near: edges.append(np.array(near))
        E = np.vstack(edges); size = n + a.max() + b.max() + 2
        data["dup_group"] = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(size, size)), directed=False)[1][:n]
        print("baris:", n, "| grup independen:", data.dup_group.nunique(), "| pasangan near-duplicate:", len(near) if NEAR_DUP else "nonaktif")
    ''')
    book.code(r'''
        classes = np.array(sorted(data[LABEL_COL].unique())); C = len(classes)
        data["y"] = data[LABEL_COL].map({c: i for i, c in enumerate(classes)})
        assert C >= 3, "Notebook ini untuk klasifikasi multiclass (>= 3 kelas)."
        assert data.groupby("y").dup_group.nunique().min() >= 5, "Perlu >= 5 grup independen per kelas."
        fold = np.full(len(data), -1)
        for k, (_, idx) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(data, data.y, data.dup_group)): fold[idx] = k
        train, val, holdout = data[fold >= 2].copy(), data[fold == 1].copy(), data[fold == 0].copy()
        for x, y in [(train, val), (train, holdout), (val, holdout)]: assert set(x.dup_group).isdisjoint(y.dup_group)
        assert all(set(f.y) == set(range(C)) for f in (train, val, holdout)), "Ada kelas hilang di salah satu split."
        y_train, y_val, y_hold = train.y.to_numpy(), val.y.to_numpy(), holdout.y.to_numpy()
        mac = lambda y, p: f1_score(y, p, labels=np.arange(C), average="macro", zero_division=0)
        M = lambda y, p: {"macro_f1": mac(y, p), "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
                          "accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p)}
        show("Jumlah baris per kelas dan split", pd.DataFrame({k: f[LABEL_COL].value_counts() for k, f in [("train", train), ("validation", val), ("holdout", holdout)]}))
        print("kelas:", list(classes), "| overlap grup antar split: 0")
    ''')

    # ------------------------------------------------------------------- EDA
    book.md("""## 5. EDA: angka saja

Seluruh EDA memakai **train saja**. Validation dan holdout tidak menjadi sumber kamus, vocabulary, atau pengamatan yang memengaruhi keputusan. Tiap tabel diikuti pertanyaan yang harus dijawab pembacanya.""")
    book.md("**5.1 Kelas, panjang, duplikat.** Apakah kelas timpang (→ bobot kelas / metrik macro)? Apakah panjang teks berbeda antar kelas (→ fitur panjang bisa menipu)? Berapa persen duplikat?")
    book.code(r'''
        t = train[TEXT_COL].astype(str); nw, nc = t.str.split().str.len(), t.str.len(); k = train.safe.str.casefold().str.replace(r"\s+", " ", regex=True)
        show("Distribusi kelas · train", dist(train[LABEL_COL]))
        show("Panjang karakter dan kata", pd.DataFrame({"karakter": nc, "kata": nw}).describe(percentiles=[.01, .05, .5, .9, .95, .99, .999]).T)
        show("Panjang kata per kelas", nw.groupby(train[LABEL_COL]).describe(percentiles=[.5, .95, .99]))
        show("Duplikat train", pd.Series({"persis_ekstra": t.duplicated().sum(), "setelah_normal_ekstra": k.duplicated().sum(),
                                           "setelah_normal_%": k.duplicated().mean() * 100, "grup_teks_berulang": k.value_counts().gt(1).sum()}))
        show("Teks yang paling sering berulang", k.value_counts().head(TOP).rename("n").to_frame())
    ''')
    book.md("**5.2 Vocabulary.** Hapax (kata muncul sekali) yang besar berarti banyak ejaan unik, tanda perlunya normalisasi atau char n-gram. TTR rendah berarti banyak pengulangan.")
    book.code(r'''
        sample = t.sample(min(5000, len(t)), random_state=SEED); seqs = sample.str.casefold().str.findall(r"[^\W\d_]+")
        counts = pd.Series(Counter(w for q in seqs for w in q), dtype="int64").sort_values(ascending=False)
        docs = pd.Series(Counter(w for q in seqs for w in set(q)), dtype="int64")
        show("Vocabulary train", pd.Series({"dokumen": len(seqs), "token": counts.sum(), "unik": len(counts), "TTR_%": 100 * len(counts) / counts.sum(),
                                            "hapax": counts.eq(1).sum(), "hapax_%_tipe": 100 * counts.eq(1).mean(), "hapax_%_token": 100 * counts.eq(1).sum() / counts.sum(),
                                            "langka_<=5_%_tipe": 100 * counts.le(5).mean()}))
        show("Top kata", counts.head(TOP).rename("n").to_frame().assign(pct_token=counts.head(TOP) / counts.sum() * 100, dokumen=docs.reindex(counts.head(TOP).index),
                                                                         pct_dokumen=docs.reindex(counts.head(TOP).index) / len(seqs) * 100))
        for g in [2, 3]:
            bag = pd.Series(Counter(" ".join(q[i:i + g]) for q in seqs for i in range(len(q) - g + 1)), dtype="int64").sort_values(ascending=False)
            show(f"Top {g}-gram", bag.head(TOP).rename("n").to_frame().assign(pct=bag.head(TOP) / bag.sum() * 100))
        odd = counts.index.to_series().str.len().ge(20) | counts.index.to_series().str.count(r"([^\W\d_])\1{2,}").gt(0)
        show("Contoh hapax", counts[counts.eq(1)].head(TOP)); show("Kata panjang / huruf berulang · kandidat typo, bukan pasti", counts[odd].head(TOP))
    ''')
    book.md("**5.3 Kata yang khas per kelas.** Log-odds kata terhadap kelas lain. Kalau kata khas hanya berupa nama atau topik (bukan kata sentimen), model akan belajar topik, bukan sentimen.")
    book.code(r'''
        tok = t.str.casefold().str.findall(r"[^\W\d_]+")
        cnt = {c: Counter(w for q in tok[train[LABEL_COL] == c] for w in q) for c in classes}; tot = sum(cnt.values(), Counter())
        khas = {}
        for c in classes:
            rest = tot - cnt[c]; nc_, nr_ = sum(cnt[c].values()), sum(rest.values())
            khas[c] = pd.Series({w: np.log((v + .5) / (nc_ + 1)) - np.log((rest[w] + .5) / (nr_ + 1)) for w, v in cnt[c].items() if v >= 5}).nlargest(TOP).index.tolist()
        show("Kata paling khas per kelas · log-odds, minimal 5 kemunculan", pd.DataFrame(dict((c, pd.Series(v)) for c, v in khas.items())))
    ''')
    book.md("**5.4 Ragam noise.** Setiap pola dihitung sebagai persen baris yang memuatnya. Pola yang besar menentukan langkah preprocessing yang dipakai; pola yang berbeda antar kelas bisa jadi sinyal sekaligus sumber bias.")
    book.code(r'''
        NOISE = {"url": r"https?://|www\.", "mention": r"@\w+", "hashtag": r"#\w+", "angka": r"\d", "emoji/simbol": r"[\U0001F300-\U0001FAFF☀-➿]",
                 "kapital": r"\b[A-Z]{3,}\b", "huruf_ulang": r"([^\W\d_])\1{2,}", "baca_ulang": r"[!?.,]{2,}", "html": r"</?[a-zA-Z][^>]*>",
                 "encoding_rusak": r"�|Ã.|Â.", "tak_terlihat": r"[​﻿]", "newline": r"[\n\r]", "email": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
                 "non_ASCII": r"[^\x00-\x7f]", "html_entity": r"&(?:#\d+|[a-zA-Z]+);", "spasi_tepi": r"^\s|\s$", "spasi_ulang": r" {2,}",
                 "kontrol": r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "huruf_angka": r"\b(?=\w*[a-zA-Z])(?=\w*\d)\w+\b",
                 "slang_contoh": r"\b(?:gk|gak|nggak|ga|bgt|yg|utk|udh|klo|jg)\b", "negasi": r"\b(?:tidak|tak|bukan|belum|jangan|tanpa|gak|nggak|ga|gk)\b",
                 "tawa": r"\b(?:wk)+w*k*\b|\b(?:ha){2,}h?\b", "pertanyaan": r"\?"}
        flags = pd.DataFrame({k_: t.str.count(p, flags=0 if k_ == "kapital" else re.I).gt(0) for k_, p in NOISE.items()})
        show("Noise · % baris train, flag boleh overlap", pd.DataFrame({"n": flags.sum(), "%": flags.mean() * 100}).sort_values("n", ascending=False))
        show("Noise per kelas · %", flags.groupby(train[LABEL_COL]).mean().T * 100)
    ''')
    book.md("**5.5 Kualitas label.** Teks sama dengan label berbeda adalah konflik. Kalau label berasal dari model/leksikon (di sini ada `Confidence Score`), skor pada label itu **bukan** bukti sentimen manusia.")
    book.code(r'''
        conflict = train.groupby(k)[LABEL_COL].nunique().loc[lambda s_: s_.gt(1)]
        print("grup teks dengan konflik label:", len(conflict), "dari", k.nunique())
        show("Contoh konflik", train.loc[k.isin(conflict.index), [TEXT_COL, LABEL_COL]].head(TOP))
        if CONF_COL in train: show("Confidence pemberi label per kelas", pd.to_numeric(train[CONF_COL], errors="coerce").groupby(train[LABEL_COL]).describe(percentiles=[.1, .5, .9]))
        show("Contoh teks per kelas", train.groupby(LABEL_COL)[TEXT_COL].apply(lambda s_: s_.sample(min(3, len(s_)), random_state=SEED).str.slice(0, 160)).to_frame("teks"))
    ''')
    book.md("**5.6 Bahasa.** Estimasi bahasa pada 300 sampel. Campuran bahasa Inggris/daerah menentukan apakah model monolingual Indonesia cukup atau perlu multilingual.")
    book.code(r'''
        from langid.langid import LanguageIdentifier, model as lang_model
        from threadpoolctl import threadpool_limits
        ident = LanguageIdentifier.from_modelstring(lang_model, norm_probs=True)
        with threadpool_limits(limits=1): pred_ = sample.sample(min(300, len(sample)), random_state=SEED).map(lambda s_: ident.classify(s_[:3000]) if len(s_.split()) >= 5 else ("tidak_pasti", 0))
        show("Estimasi bahasa · % sampel (teks < 5 kata atau skor < 0,8 = tidak_pasti)", dist(pred_.map(lambda p: p[0] if p[1] >= .8 else "tidak_pasti")))
    ''')
    return book


# --------------------------------------------------------------------- ekor
def tail(book, name, linear_features=False):
    book.md("""## Evaluasi

Semua model punya skor `val_s[model]` (validation) dan `hold_s[model]` (holdout) berbentuk `(baris, kelas)`: log-probabilitas untuk LR/MLP/BERT, margin untuk SVM. Urutan evaluasi: **pilih di validation → bekukan → buka holdout sekali**.""")
    book.md("### Pilih model di validation\n\nMetrik utama **macro-F1** (tiap kelas berbobot sama). Ensemble = rata-rata log-probabilitas tiga model probabilistik terbaik. `train_f1` jauh di atas validation berarti overfit.")
    book.code(r'''
        prob_models = [m for m in val_s if "svm" not in m and m != "majority"]
        top3 = pd.Series({m: mac(y_val, val_s[m].argmax(1)) for m in prob_models}).nlargest(3).index.tolist()
        for S in (val_s, hold_s): S["ens_top3"] = np.mean([S[m] for m in top3], axis=0)
        leader = pd.DataFrame({m: M(y_val, S.argmax(1)) for m, S in val_s.items()}).T.join(pd.DataFrame(rec).set_index("model"))
        leader = leader.sort_values("macro_f1", ascending=False); best = leader.index[0]
        show("Validation · semua model", leader)
        print("ensemble dari:", top3, "| pemenang validation:", best)
    ''')
    book.md("### Post-processing: bias per kelas\n\nMenambah bias kecil pada skor tiap kelas untuk mengangkat kelas yang jarang diprediksi. Bias dicari dengan coordinate ascent, dinilai **cross-fit dua fold** pada validation (dicari di satu fold, diuji di fold lain), dan hanya dipakai bila naik lebih dari `MIN_POST_GAIN`.")
    book.code(r'''
        grid = np.array([0, .25, -.25, .5, -.5, 1, -1]); S = val_s[best]; cross = np.zeros(len(val), int)
        folds = list(StratifiedGroupKFold(2, shuffle=True, random_state=SEED).split(val, y_val, val.dup_group)) + [(np.arange(len(val)),) * 2]
        for tune, check in folds:
            bias = np.zeros(C)
            for _ in range(2):
                for c in range(C): bias[c] = grid[np.argmax([mac(y_val[tune], (S[tune] + np.where(np.arange(C) == c, g, bias)).argmax(1)) for g in grid])]
            if len(check) < len(val): cross[check] = (S[check] + bias).argmax(1)
        raw_f1, cross_f1 = mac(y_val, S.argmax(1)), mac(y_val, cross); use_post = cross_f1 > raw_f1 + MIN_POST_GAIN
        show("Bias per kelas · cross-fit di validation", pd.Series({"macro_f1_asli": raw_f1, "macro_f1_cross_fit": cross_f1, "bias_dipakai": use_post}))
        print("bias akhir per kelas:", dict(zip(classes, bias.round(2))))
        selection = {"model": best, "post_processing": bool(use_post), "bias": bias.tolist(), "kelas": classes.tolist()}
        Path(OUT, "selection.json").write_text(json.dumps(selection, indent=2)); print("dibekukan sebelum holdout:", selection)
    ''')
    book.md("### Holdout: dibuka sekali\n\nPemilihan tidak diubah berdasarkan tabel ini. Selisih validation → holdout mengukur seberapa optimistis seleksi di validation. Interval kepercayaan memakai bootstrap per grup duplikat.")
    book.code(r'''
        hold = pd.DataFrame({m: M(y_hold, S_.argmax(1)) for m, S_ in hold_s.items()}).T
        hold["val_macro_f1"] = leader.macro_f1; hold["selisih_val_hold"] = hold.val_macro_f1 - hold.macro_f1
        show("Holdout · semua model, urut validation", hold.loc[leader.index])
        final = (hold_s[best] + (bias if use_post else 0)).argmax(1)
        rng, gid = np.random.default_rng(SEED), holdout.dup_group.to_numpy(); ug = np.unique(gid); by = {g: np.flatnonzero(gid == g) for g in ug}
        boot = [mac(y_hold[i], final[i]) for i in (np.concatenate([by[g] for g in rng.choice(ug, len(ug))]) for _ in range(300))]
        show("Pilihan final · holdout", pd.DataFrame([{"model": best, "post_processing": bool(use_post), **M(y_hold, final),
                                                       "macro_f1_CI95_bawah": np.percentile(boot, 2.5), "macro_f1_CI95_atas": np.percentile(boot, 97.5)}]))
    ''')
    book.md("### Per kelas dan confusion\n\nMacro-F1 yang bagus bisa menutupi satu kelas yang gagal. Baca precision / recall / support per kelas, lalu pasangan kelas yang paling sering tertukar.")
    book.code(r'''
        rep = classification_report(y_hold, final, labels=np.arange(C), target_names=classes, output_dict=True, zero_division=0)
        show("Precision / recall / F1 / support · model final", pd.DataFrame(rep).T)
        show("F1 per kelas · semua model", pd.DataFrame({m: f1_score(y_hold, S_.argmax(1), labels=np.arange(C), average=None, zero_division=0) for m, S_ in hold_s.items()}, index=classes).T)
        cm = pd.DataFrame(confusion_matrix(y_hold, final, labels=np.arange(C)), index=classes, columns=classes)
        show("Confusion · baris = aktual, kolom = prediksi", cm); show("Confusion · % per kelas aktual", cm.div(cm.sum(1), axis=0) * 100)
        off = cm.values.copy(); np.fill_diagonal(off, 0); i, j = np.unravel_index(off.argmax(), off.shape)
        print(f"pasangan paling tertukar: aktual '{classes[i]}' diprediksi '{classes[j]}' = {off[i, j]} baris ({off[i, j] / cm.values[i].sum():.1%} dari kelas itu)")
    ''')
    book.md("### Di mana model lemah?\n\nPerforma dipecah menurut kondisi teks. Slice dengan `n` kecil tidak bisa disimpulkan; bandingkan `error_%` dengan error rata-rata. F1 slice dihitung hanya atas kelas yang hadir di slice itu.")
    book.code(r'''
        an = holdout.copy(); an["prediksi"] = classes[final]; an["salah"] = final != y_hold
        an["panjang_kata"] = pd.cut(an.safe.str.split().str.len(), [-1, 5, 15, 40, 100, np.inf])
        for kname in ["negasi", "slang_contoh", "url", "emoji/simbol", "huruf_ulang", "pertanyaan", "tawa"]: an[kname] = an[TEXT_COL].astype(str).str.count(NOISE[kname], flags=re.I).gt(0)
        if CONF_COL in an: an["confidence_label"] = pd.cut(pd.to_numeric(an[CONF_COL], errors="coerce"), [-np.inf, .6, .8, .95, np.inf])
        cols = ["panjang_kata", "negasi", "slang_contoh", "url", "emoji/simbol", "huruf_ulang", "pertanyaan", "tawa"] + (["confidence_label"] if CONF_COL in an else [])
        rows = [{"slice": c, "nilai": str(v), "n": len(f), "error_%": f.salah.mean() * 100, "accuracy": accuracy_score(f.y, classes.searchsorted(f.prediksi)),
                 "macro_f1_kelas_hadir": f1_score(f.y, classes.searchsorted(f.prediksi), labels=np.unique(f.y), average="macro", zero_division=0)}
                for c in cols for v, f in an.groupby(c, observed=True, dropna=False)]
        slices = pd.DataFrame(rows); print("error rata-rata holdout:", round(an.salah.mean() * 100, 1), "%")
        show("Performa per kondisi teks", slices)
    ''')
    book.md("**Salah prediksi paling yakin.** Baca teksnya: kalau labelnya yang ambigu atau salah, itu batas atas yang bisa dicapai model, bukan kelemahan model.")
    book.code(r'''
        an["skor_model"] = hold_s[best].max(1) - np.sort(hold_s[best], axis=1)[:, -2]       # selisih skor tertinggi dan kedua
        show("Salah prediksi dengan margin terbesar", an[an.salah].sort_values("skor_model", ascending=False)[[TEXT_COL, LABEL_COL, "prediksi", "skor_model"] + ([CONF_COL] if CONF_COL in an else [])].head(TOP))
        weak = pd.DataFrame(rep).T.loc[classes].sort_values("f1-score").iloc[0]
        print("kelas F1 terendah:", weak.name, "| F1:", round(weak["f1-score"], 3), "| support:", int(weak.support))
        show("Slice error tertinggi, n >= 20", slices[slices.n.ge(20)].sort_values("error_%", ascending=False).head(5))
    ''')
    if linear_features:
        book.md("**Fitur pendorong tiap kelas** (koefisien LR TF-IDF): menunjukkan apakah model belajar kata sentimen atau kata topik.")
        book.code(r'''
            lr_name = next(m for m in fitted if m.startswith("tfidf_word+lr")); feats = np.array(rep_vec["tfidf_word"].get_feature_names_out())
            show("Kata/n-gram pendorong tiap kelas", pd.DataFrame({c: feats[np.argsort(fitted[lr_name].coef_[i])[-10:][::-1]] for i, c in enumerate(classes)}))
        ''')
    book.md("### Ringkasan keputusan dan keluaran\n\nTemuan → keputusan, dibuat dari angka run ini. File kecil disimpan ke `hasil/` untuk dibaca ulang tanpa menjalankan notebook.")
    book.code(r'''
        decisions = pd.DataFrame([
            {"temuan": "grup duplikat", "bukti": f"{data.dup_group.nunique()} grup dari {len(data)} baris", "tindakan": "grup dibentuk sebelum split, overlap = 0"},
            {"temuan": "kelas terlemah", "bukti": f"{weak.name}: F1 {weak['f1-score']:.3f}, support {int(weak.support)}", "tindakan": "baca confusion dan contoh salah sebelum eksperimen berikutnya"},
            {"temuan": "pasangan tertukar", "bukti": f"{classes[i]} → {classes[j]}: {off[i, j]} baris", "tindakan": "periksa batas label antar dua kelas ini"},
            {"temuan": "model pilihan", "bukti": best, "tindakan": "dipilih di validation; holdout hanya laporan"},
            {"temuan": "post-processing", "bukti": f"cross-fit {cross_f1:.3f} vs asli {raw_f1:.3f}", "tindakan": "dipakai" if use_post else "tidak dipakai (kenaikan < MIN_POST_GAIN)"},
            {"temuan": "ketidakpastian", "bukti": f"CI95 holdout {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}", "tindakan": "selisih antar model kecil dari lebar CI tidak bisa dianggap beda"}])
        show("Temuan → keputusan", decisions)
        leader.to_csv(Path(OUT, "validation_models.csv")); hold.to_csv(Path(OUT, "holdout_models.csv")); slices.to_csv(Path(OUT, "error_slices.csv"), index=False)
        an[["row_id", TEXT_COL, LABEL_COL, "prediksi", "salah", "skor_model"]].to_csv(Path(OUT, "holdout_predictions.csv"), index=False)
        pd.DataFrame(rep).T.to_csv(Path(OUT, "per_class.csv"))
        manifest = {"notebook": "''' + name + r'''", "baris": len(data), "split": {"train": len(train), "validation": len(val), "holdout": len(holdout)},
                    "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": SEED, "final": {"model": best, **M(y_hold, final)},
                    "detik": time.perf_counter() - START, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        Path(OUT, "run_manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2))
    ''')
    book.md("""## Batas interpretasi

- Label contoh adalah pseudo-label; skor terhadap label ini **bukan** akurasi sentimen manusia.
- Holdout tunggal memberi satu estimasi; baca bersama interval kepercayaan. Selisih antar model yang lebih kecil dari lebar interval tidak bisa dianggap berbeda.
- Kamus preprocessing (`SLANG`, `PROTECTED`, `VOCAB`) disusun untuk komentar MBG; tinjau ulang untuk domain lain.
- Referensi: [IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased), [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [group cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).""")
    return book


# --------------------------------------------------------------- penyelesaian
def finalize(book, dest_name, slug, title, gpu):
    for c in book.cells:
        if c.cell_type == "code":
            ast.parse(c.source)
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    notebook = nb.v4.new_notebook(cells=book.cells, metadata=meta)
    nb.validate(notebook)
    dest = FOLDER / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    nb.write(notebook, dest)
    kernel = Path(tempfile.gettempdir()) / f"gk_{slug}"
    kernel.mkdir(parents=True, exist_ok=True)
    payload = base64.b64encode(nb.writes(notebook).encode()).decode()
    runner = f'''import base64, pathlib, os
import papermill as pm
os.environ['PYTHONHASHSEED']='42'
src = pathlib.Path('/kaggle/working/_source.ipynb')
src.write_bytes(base64.b64decode({payload!r}))
try:
    pm.execute_notebook(str(src), '/kaggle/working/{dest_name}', kernel_name='python3', cwd='/kaggle/working', progress_bar=False, log_output=True)
finally:
    src.unlink(missing_ok=True)
'''
    nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)], metadata=meta), kernel / "runner.ipynb")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ivanwllm/{slug}", "title": title, "code_file": "runner.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_internet": True, "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=2))
    print("ditulis:", dest, "| sel:", len(book.cells), "| baris kode:", sum(len(c.source.splitlines()) for c in book.cells if c.cell_type == "code"), "| kernel:", kernel)
    return kernel
