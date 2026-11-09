"""Author clustering review notebook; execution only on Kaggle (papermill, dua dataset)."""
from pathlib import Path
from textwrap import dedent
import ast, base64, json, tempfile, hashlib
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'a_tanpa_label/03_clustering/03_clustering_review.ipynb'
UNIVERSAL = ROOT / 'preprocessing_universal/stepwise.py'
CELLS = []


def md(s): CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s): CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def part(title, note, *sources):
    md('## ' + title + '\n\n' + note)
    for source in sources: code(source)


md(r'''
# Clustering · versi review
**Pilot satu kasus: EDA angka → preprocessing per representasi → 5 representasi × beberapa algoritma → post-processing → evaluasi kelemahan.**

Run All pada **Kaggle GPU**. Semua angka berasal dari data asli, tanpa grafik otomatis.

| Representasi | Preprocessing | Algoritma yang diuji |
|---|---|---|
| TF-IDF kata 1-2 gram | BoW agresif: normalisasi, tanpa angka/tanda baca/stopword | KMeans, NMF |
| LSA (TF-IDF + SVD 100) | sama | KMeans, Agglomerative |
| Word2Vec rata-rata, dilatih di data ini | sama | KMeans |
| IndoBERT mean pooling, tanpa fine-tuning | safe + placeholder, teks utuh | KMeans |
| multilingual-e5 sentence embedding | safe + placeholder, teks utuh | KMeans, Agglomerative, UMAP+HDBSCAN |
| Pembanding | Acak dengan ukuran cluster sama, LDA dari hitungan kata | batas bawah dan topic model klasik |

**Aturan utama: tidak ada label saat memilih.** Jumlah cluster, representasi, preprocessing, dan post-processing dipilih dari kriteria tanpa label: silhouette di ruang e5, stabilitas bootstrap (ARI), dan ukuran cluster. NPMI hanya pemecah seri. Data contoh punya kategori asli, tapi kolom itu disimpan terpisah sebagai `audit` dan baru dipakai untuk **menilai** setelah keputusan dibuat. Dari situ terlihat kriteria tanpa label mana yang benar-benar searah dengan kebenaran.

**Data:** `DATASET="news"` memakai [judul berita Indonesia](https://www.kaggle.com/datasets/ibamibrahim/indonesian-news-title), dengan kategori sebagai audit. `DATASET="mbg"` memakai [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program) tanpa label topik, yaitu kondisi yang mirip lomba. Kolom sentimen tidak dipakai, karena sentimen bukan topik.
''')

part('00 · Konfigurasi', 'Sel ini menjadi parameter papermill. Untuk data baru, tambah satu preset: nama file, kolom teks, dan kolom audit. Isi `None` kalau labelnya tidak ada.', r'''
DATASET = "news"
SEED, N_ROWS, TOP = 42, 5000, 12
K_GRID = [5, 6, 8, 10, 12, 15]
BOOTSTRAP, SUBSAMPLE = 5, .8          # stabilitas: 5 KMeans pada subsample 80%, ARI antar pasangan
MAX_SHARE = .40                         # cluster terbesar maksimal 40% baris
MIN_DF, DF_MAX, MIN_CLUSTER_SHARE = 3, .25, .01
MAX_LEN, BATCH = 128, 64
PRESETS = {"news": dict(file="indonesian-news-title.csv", text="title", audit="category"),
           "mbg": dict(file="comment_sentiment.csv", text="Comment", audit=None)}
''', r'''
from pathlib import Path
FILE, TEXT_COL, AUDIT_COL = PRESETS[DATASET]["file"], PRESETS[DATASET]["text"], PRESETS[DATASET]["audit"]
OUT = Path(f"/kaggle/working/hasil_clustering_{DATASET}")
print("dataset:", DATASET, "| teks:", TEXT_COL, "| audit:", AUDIT_COL or "tidak ada")
''')

part('01 · Lingkungan Kaggle', 'Paket tambahan dipasang di Kaggle. Model diambil dari paket offline kalau tersedia sebagai input, dan dari Hugging Face kalau tidak.', r'''
import sys, os, subprocess
assert Path("/kaggle/input").exists(), "Jalankan notebook ini di Kaggle."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn>=0.5,<0.6", "gensim>=4.4,<5", "ftfy>=6.3,<7",
                "emoji>=2.14,<3", "datasketch>=1.6,<3", "langid==1.1.6", "Sastrawi==1.0.1"], check=True)
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1", USE_TF="0", USE_FLAX="0")
''', r'''
import json, time, random, re, html, unicodedata, itertools, hashlib, warnings
from collections import Counter
import numpy as np, pandas as pd, torch, emoji, ftfy
from IPython.display import display
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT.mkdir(exist_ok=True)
INPUT = Path("/kaggle/input")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak ada")
RUN_START = time.perf_counter()
''', r'''
def show(title, table):
    print("\n" + title)
    display((table.to_frame() if isinstance(table, pd.Series) else table).round(3))

def distribution(values):
    n = values.value_counts(dropna=False)
    return pd.DataFrame({"n": n, "%": n / (len(values) or np.nan) * 100})

pd.set_option("display.max_columns", None, "display.max_colwidth", 110, "display.width", 200)
''')

part('02 · Muat data mentah & audit sebelum drop', 'Missing dan teks invalid diaudit sebelum dibuang. Kolom audit langsung dipisah dari data, dan tidak dipakai sampai bagian 09.', r'''
path = next(INPUT.rglob(FILE))
raw = pd.read_csv(path)
assert TEXT_COL in raw, f"Kolom tersedia: {raw.columns.tolist()}"
show("Skema sebelum pembersihan", pd.DataFrame({"tipe": raw.dtypes.astype(str), "missing": raw.isna().sum(),
     "missing_%": raw.isna().mean() * 100, "unik_non_null": raw.nunique(dropna=True)}))
text = raw[TEXT_COL]
quality = pd.DataFrame({"teks_null": text.isna(), "teks_non_string": text.notna() & ~text.map(lambda x: isinstance(x, str)),
    "teks_kosong": text.fillna("").astype(str).str.strip().eq(""),
    "teks_pseudo_null": text.fillna("").astype(str).str.strip().str.casefold().isin(["nan", "null", "none", "n/a", "-"])})
if AUDIT_COL: quality["audit_null"] = raw[AUDIT_COL].isna()
show("Kualitas input · flag dapat overlap", pd.DataFrame({"n": quality.sum(), "%": quality.mean() * 100}))
''', r'''
data = raw.loc[~quality.any(axis=1)].reset_index(names="row_id")
if N_ROWS and len(data) > N_ROWS: data = data.sample(N_ROWS, random_state=SEED).reset_index(drop=True)
data["_audit"] = data[AUDIT_COL].astype(str) if AUDIT_COL else None
print("File:", path.name, "| baris mentah:", len(raw), "| invalid:", int(quality.any(axis=1).sum()), "| baris pilot:", len(data))
''')

part('03 · EDA wajib clustering · angka', 'Pertanyaan EDA untuk clustering: berapa teks yang identik (menggelembungkan cluster), seberapa pendek (sinyal sedikit), seberapa jarang kosakatanya (TF-IDF jadi sparse), kata apa yang ada di hampir semua dokumen (menyatukan semua cluster), noise apa yang perlu dibuang, bahasa apa yang dipakai, dan apakah ada template berulang. Setiap angka menghasilkan keputusan di 03.6.', r'''
t = data[TEXT_COL].astype(str)
key = t.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
words = t.str.split().str.len()

def near_duplicate_count(keys, threshold=.9):
    from datasketch import MinHash, MinHashLSH
    index, grams_of, linked = MinHashLSH(threshold=threshold, num_perm=64), {}, set()
    for i, k in enumerate(dict.fromkeys(keys)):
        if len(k) < 20: continue
        grams = {k[j:j + 5] for j in range(len(k) - 4)}
        signature = MinHash(num_perm=64, seed=SEED); signature.update_batch([g.encode() for g in sorted(grams)])
        for other in index.query(signature):
            if len(grams & grams_of[other]) / len(grams | grams_of[other]) >= threshold: linked |= {other, str(i)}
        index.insert(str(i), signature); grams_of[str(i)] = grams
    return len(linked)

show("3.1 Ukuran, duplikat, teks pendek", pd.Series({"baris": len(t), "duplikat_persis": t.duplicated().sum(),
    "duplikat_normal": key.duplicated().sum(), "duplikat_normal_%": key.duplicated().mean() * 100,
    "grup_duplikat": key.value_counts().gt(1).sum(), "teks_unik_dalam_near_dup": near_duplicate_count(key.tolist()),
    "teks_<3_kata_%": words.lt(3).mean() * 100, "teks_<5_kata_%": words.lt(5).mean() * 100, "teks_>100_kata_%": words.gt(100).mean() * 100}))
show("Panjang karakter/kata", pd.DataFrame({"karakter": t.str.len(), "kata": words}).describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)
''', r'''
seqs = t.str.casefold().str.findall(r"[^\W\d_]+")
counts = pd.Series(Counter(w for s in seqs for w in s), dtype="int64").sort_values(ascending=False)
docfreq = pd.Series(Counter(w for s in seqs for w in set(s)), dtype="int64").reindex(counts.index) / len(seqs)
show("3.2 Kosakata", pd.Series({"token": counts.sum(), "tipe_unik": len(counts), "TTR_%": 100 * len(counts) / counts.sum(),
    "hapax_%_tipe": 100 * counts.eq(1).mean(), "hapax_%_token": 100 * counts.eq(1).sum() / counts.sum(),
    "tipe_df<MIN_DF_%": 100 * (docfreq * len(seqs)).lt(MIN_DF).mean(), "kata_df>DF_MAX": int(docfreq.gt(DF_MAX).sum()),
    "token_per_dok_median": seqs.str.len().median(), "dok_tanpa_kata": int(seqs.str.len().eq(0).sum())}))
show("Top words · n dan % dokumen", pd.DataFrame({"n": counts.head(TOP), "%_dokumen": 100 * docfreq.head(TOP)}))
bigrams = pd.Series(Counter(" ".join(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)), dtype="int64").nlargest(TOP)
show("Top bigram", bigrams)
show("Contoh hapax · kandidat typo, nama, atau slang", counts[counts.eq(1)].sample(min(TOP, int(counts.eq(1).sum())), random_state=SEED))
''', r'''
NOISE = {"url": r"https?://|www\.", "html": r"</?[a-zA-Z][^>]*>", "html_entity": r"&(?:#\d+|[a-zA-Z]+);",
    "mention": r"@\w+", "hashtag": r"#\w+", "emoji": r"[\U0001F300-\U0001FAFF☀-➿]", "angka": r"\d",
    "huruf_ulang": r"([^\W\d_])\1{2,}", "tanda_baca_ulang": r"[!?.,]{2,}", "kapital_semua": r"\b[A-Z]{3,}\b",
    "non_ASCII": r"[^\x00-\x7f]", "encoding_rusak": r"�|Ã.|Â.", "invisible": r"[​﻿]",
    "slang_contoh": r"\b(?:gk|gak|ga|bgt|yg|utk|udh|klo|jg|tdk)\b", "tawa": r"\b(?:wk){2,}\w*|\b(?:ha){2,}\b"}
flags = pd.DataFrame({name: t.str.contains(p, flags=0 if name == "kapital_semua" else re.I) for name, p in NOISE.items()})
show("3.3 Variasi noise · % baris", pd.DataFrame({"n": flags.sum(), "%": flags.mean() * 100}).sort_values("n", ascending=False))
''', r'''
from langid.langid import LanguageIdentifier, model as language_model
identifier = LanguageIdentifier.from_modelstring(language_model, norm_probs=True)
language = t.sample(min(500, len(t)), random_state=SEED).map(lambda s: identifier.classify(s[:2000]))
show("3.4 Estimasi bahasa · sampel 500, yakin >= 0.8", distribution(language.map(lambda p: p[0] if p[1] >= .8 else "tidak_pasti")))
prefix = seqs.map(lambda s: " ".join(s[:3]))
top_prefix = prefix[prefix.str.count(" ").ge(2)].value_counts().head(TOP)
show("3.5 Template · awalan 3 kata yang paling sering", top_prefix.to_frame("n").assign(**{"%": 100 * top_prefix / len(t)}))
''', r'''
if AUDIT_COL: show("Kategori audit · hanya untuk penilaian, tidak memilih apa pun", distribution(data["_audit"]))
short, dup = words.lt(5).mean(), key.duplicated().mean()
lang_share = language.map(lambda p: p[0] if p[1] >= .8 else "?").value_counts(normalize=True)
decisions = pd.DataFrame([
    {"temuan": "duplikat normal", "angka": f"{dup:.1%}", "keputusan": "post-processing: teks identik dipaksa satu cluster" if dup > 0 else "tidak perlu"},
    {"temuan": "teks < 5 kata", "angka": f"{short:.1%}", "keputusan": "BoW lemah untuk teks pendek, embedding diuji" if short > .2 else "BoW masih layak"},
    {"temuan": "hapax % tipe", "angka": f"{counts.eq(1).mean():.1%}", "keputusan": f"TF-IDF min_df={MIN_DF}; stemming diuji di 04.2"},
    {"temuan": f"kata df > {DF_MAX:.0%}", "angka": str(int(docfreq.gt(DF_MAX).sum())), "keputusan": "dibuang dari BoW sebagai stopword domain"},
    {"temuan": "noise teknis tertinggi", "angka": f"{flags.mean().idxmax()} {flags.mean().max():.1%}", "keputusan": "dibersihkan di safe; placeholder untuk url/mention"},
    {"temuan": "bahasa non-id/tidak pasti", "angka": f"{1 - lang_share.get('id', 0):.1%}", "keputusan": "embedding multilingual e5 ikut diuji"},
    {"temuan": "template awalan terbesar", "angka": f"{top_prefix.iloc[0] / len(t):.1%}" if len(top_prefix) else "0", "keputusan": "cek cluster template di 09.3"}])
show("3.6 Temuan → keputusan", decisions)
''')

md('## 04 · Preprocessing dari universal\n\nFungsi berikut diambil langsung dari `preprocessing_universal/stepwise.py`. Clustering butuh dua jenis input. **BoW** (TF-IDF, LDA, NMF, Word2Vec) perlu agresif, karena kata umum, angka, dan tanda baca justru menyatukan semua dokumen. **Embedding** butuh kalimat utuh, jadi hanya dibersihkan secara teknis, lalu URL dan mention diganti placeholder.')
u = UNIVERSAL.read_text(encoding='utf-8'); lines = u.splitlines(); tree = ast.parse(u)
code(u.split('# %% resources\n', 1)[1].split('# %% validate', 1)[0])
functions = {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno]) for node in tree.body if isinstance(node, ast.FunctionDef)}
for names in [['valid_text', 'repair_encoding', 'normalize_unicode', 'remove_html', 'decode_html_entities', 'remove_invisible', 'normalize_whitespace'],
              ['replace_urls', 'replace_emails', 'replace_phones', 'replace_mentions', 'segment_hashtags', 'lowercase'],
              ['normalize_slang', 'normalize_elongation', 'normalize_laughter', 'normalize_reduplication'],
              ['remove_emoji', 'remove_emoticons', 'remove_numbers', 'remove_punctuation', 'reduce_repeated_words']]:
    code('\n\n'.join(functions[name] for name in names))
code('from functools import lru_cache\n' + '\n'.join(lines[[n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_stemmer'][0] - 2:
     [n.end_lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stem_text'][0]]))
md('**4.1 Empat versi teks.** `safe` hanya perbaikan teknis. `embed` = safe + placeholder. `normal` = embed + lowercase dan normalisasi kata. `bow` = normal tanpa emoji, angka, tanda baca, kata berulang, stopword umum, dan stopword domain (df > `DF_MAX`). `bow_stem` = bow + stemming Sastrawi.')
code(r'''
def stem_series(series):
    """Stem hanya kata yang lolos MIN_DF, 4 proses paralel. Kata langka tetap, toh dibuang TF-IDF."""
    # forkserver: proses anak tidak mewarisi thread CUDA/torch, yang bisa membuat fork macet.
    import multiprocessing as mp
    Path("stem_worker.py").write_text("from Sastrawi.Stemmer.StemmerFactory import StemmerFactory\n_s = None\n"
        "def stem(w):\n    global _s\n    _s = _s or StemmerFactory().create_stemmer()\n    return _s.stem(w)\n")
    if "." not in sys.path: sys.path.insert(0, ".")
    import stem_worker
    vocab = pd.Series(Counter(w for s in series.str.split() for w in set(s)))
    frequent = [w for w in vocab[vocab >= MIN_DF].index if w not in PROTECTED | NEGATIONS]
    with mp.get_context("forkserver").Pool(4) as pool:
        stems = dict(zip(frequent, pool.map(stem_worker.stem, frequent, chunksize=256)))
    print(f"stemming {len(frequent)} kata dari {len(vocab)} kata unik")
    return series.map(lambda t: " ".join(stems.get(w, w) for w in t.split()))
def chain(text, steps):
    for step in steps: text = step(text)
    return text
def apply(series, steps):
    cache = {s: chain(s, steps) for s in series.unique()}
    return series.map(cache)
SAFE = [valid_text, repair_encoding, normalize_unicode, lambda s: remove_html(s) if "<" in s else s,
        decode_html_entities, remove_invisible, normalize_whitespace]
PLACEHOLDER = [replace_urls, replace_emails, replace_phones, replace_mentions, normalize_whitespace]
NORMAL = [segment_hashtags, lowercase, normalize_slang, normalize_elongation, normalize_laughter, normalize_reduplication]
CLEAN = [remove_emoji, remove_emoticons, remove_numbers, remove_punctuation, reduce_repeated_words, normalize_whitespace]
STOP = STOPWORDS | set("""akan ada adalah agar aja aku apa atau bahwa banyak baru bila bisa buat bukan cuma dah dalam dapat demi dgn
    dong gitu harus hal hanya ini itu jadi jika juga kalau kami kamu karena kata ke kenapa kok kini lagi lah lain lalu lebih
    loh maka masih mau mereka meski mungkin nah namun nanti nya oleh pada para pernah pun saat saja sama sampai sangat
    sebagai sedang sejak sekarang semua seperti setelah sih sudah supaya tapi telah tentang terhadap tersebut tetapi
    untuk usai yaitu yang ya url user email phone wkwk haha""".split())
data["safe"] = apply(data[TEXT_COL], SAFE)
data["embed"] = apply(data.safe, PLACEHOLDER)
data["normal"] = apply(data.embed, NORMAL)
bow = apply(data.normal, CLEAN).str.split()
df_bow = pd.Series(Counter(w for s in bow for w in set(s))) / len(bow)
DOMAIN_STOP = set(df_bow[df_bow > DF_MAX].index)
show(f"Stopword domain · df > {DF_MAX:.0%}, dibuang dari BoW", df_bow[df_bow > DF_MAX].sort_values(ascending=False).rename("df"))
data["bow"] = bow.map(lambda s: " ".join(w for w in s if w not in STOP | DOMAIN_STOP and len(w) > 1))
data["bow_stem"] = stem_series(data.bow)
''')
code(r'''
views = ["safe", "embed", "normal", "bow", "bow_stem"]
show("Audit preprocessing", pd.DataFrame({v: {"berubah_dari_sebelumnya_%": 100 * data[v].ne(data[views[max(i - 1, 0)]] if i else data[TEXT_COL]).mean(),
    "kosong_%": 100 * data[v].str.strip().eq("").mean(), "kata_median": data[v].str.split().str.len().median(),
    "kosakata": len(set(" ".join(data[v]).split()))} for i, v in enumerate(views)}).T)
show("Before/after", data[[TEXT_COL, "embed", "bow", "bow_stem"]].sample(5, random_state=SEED))
print("Baris BoW kosong tetap diberi cluster; di TF-IDF baris itu menjadi vektor nol.")
''')

part('05 · Fungsi evaluasi tanpa label', '''Semua kriteria di sini **tidak memakai label**.
- **Stabilitas (ARI):** KMeans diulang pada subsample 80% dengan seed berbeda. Kalau hasilnya berubah-ubah, cluster itu artefak.
- **NPMI:** apakah 10 kata kunci tiap cluster benar-benar muncul bersama di dokumen. Dihitung pada korpus `bow` yang sama untuk semua model, jadi bisa dibandingkan antar representasi.
- **Diversity:** porsi kata kunci yang unik ke satu cluster.
- **Silhouette, Davies-Bouldin, Calinski-Harabasz:** bergantung ruang vektornya, jadi hanya adil dibandingkan dalam satu representasi. Silhouette juga dihitung di ruang e5 bersama, tetapi cara ini menguntungkan model berbasis e5.

`audit_*` dihitung terakhir, dan hanya kalau kolom audit ada.''', r'''
reference = CountVectorizer(binary=True, min_df=MIN_DF, token_pattern=r"\S+").fit(data.bow)
DOC = reference.transform(data.bow).tocsc(); VOCAB_REF = reference.vocabulary_; N_DOC = DOC.shape[0]

def keywords(labels, texts=None, n=10):
    texts = data.bow.to_numpy() if texts is None else texts
    clusters = [c for c in np.unique(labels) if c >= 0]
    vectorizer = CountVectorizer(token_pattern=r"\S+").fit([" ".join(texts[labels == c]) for c in clusters])
    tf = vectorizer.transform([" ".join(texts[labels == c]) for c in clusters]).toarray().astype(float)
    weight = tf * np.log1p(tf.sum() / len(clusters) / np.maximum(tf.sum(0), 1))   # c-TF-IDF
    names = vectorizer.get_feature_names_out()
    return {c: list(names[np.argsort(-weight[i])[:n]]) for i, c in enumerate(clusters)}

def npmi(words):
    ids = [VOCAB_REF[w] for w in words if w in VOCAB_REF]
    scores = []
    for a, b in itertools.combinations(ids, 2):
        both = DOC[:, a].multiply(DOC[:, b]).sum() / N_DOC
        pa, pb = DOC[:, a].sum() / N_DOC, DOC[:, b].sum() / N_DOC
        scores.append(-1. if both == 0 else np.log(both / (pa * pb)) / -np.log(both))
    return float(np.mean(scores)) if scores else np.nan
''', r'''
def audit_scores(labels):
    truth = data["_audit"].to_numpy()
    table = pd.crosstab(labels, truth).to_numpy()
    rows, cols = linear_sum_assignment(-table)
    h, c, v = homogeneity_completeness_v_measure(truth, labels)
    return {"audit_NMI": normalized_mutual_info_score(truth, labels), "audit_ARI": adjusted_rand_score(truth, labels),
            "audit_homogeneity": h, "audit_completeness": c, "audit_acc_hungarian": table[rows, cols].sum() / len(labels),
            "audit_purity": table.max(1).sum() / len(labels)}

def stability(X, k, return_runs=False):
    rng, runs = np.random.default_rng(SEED), []
    for b in range(BOOTSTRAP):
        sub = rng.choice(X.shape[0], int(SUBSAMPLE * X.shape[0]), replace=False)
        runs.append(KMeans(k, n_init=1, random_state=SEED + b).fit(X[sub]).predict(X))
    score = float(np.mean([adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)]))
    return (score, runs) if return_runs else score

rows = []
def evaluate(name, labels, X=None, **extra):
    labels = np.asarray(labels); valid = labels >= 0
    sizes = pd.Series(labels[valid]).value_counts(normalize=True)
    keys = keywords(labels); top = [w for ws in keys.values() for w in ws]
    row = {"model": name, **extra, "k": len(sizes), "noise_%": 100 * (~valid).mean(), "max_share_%": 100 * sizes.max(),
           "min_share_%": 100 * sizes.min(), "NPMI": np.nanmean([npmi(ws) for ws in keys.values()]),
           "diversity": len(set(top)) / max(len(top), 1)}
    pick = np.random.default_rng(SEED).choice(np.where(valid)[0], min(2000, int(valid.sum())), replace=False)
    if len(sizes) > 1 and len(np.unique(labels[pick])) > 1:
        if X is not None:
            row["silhouette_own"] = silhouette_score(X[pick], labels[pick], metric="cosine")
            if not hasattr(X, "tocsr"):
                row["DB_own"] = davies_bouldin_score(X[pick], labels[pick]); row["CH_own"] = calinski_harabasz_score(X[pick], labels[pick])
        if "E5" in globals(): row["silhouette_e5"] = silhouette_score(E5[pick], labels[pick], metric="cosine")
    if AUDIT_COL: row.update(audit_scores(labels))
    rows.append(row)
    return row
print("Fungsi evaluasi siap. Kosakata referensi NPMI:", len(VOCAB_REF))
''')

part('06 · Representasi teks', 'Lima representasi, masing-masing memakai input yang sesuai. TF-IDF, SVD, dan Word2Vec hanya belajar dari teks, tanpa label. Model pretrained tidak di-fine-tune, karena clustering tidak punya target untuk melatihnya.', r'''
def locate(name, hf_id):
    found = list(INPUT.glob(f"**/models/{name}/config.json"))
    return str(found[0].parent) if found else hf_id

def embed(model_id, texts):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id); model = AutoModel.from_pretrained(model_id).cuda().half().eval()
    order = np.argsort([len(x) for x in texts]); parts = []; cut = 0
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            batch = [texts[j] for j in order[i:i + BATCH]]
            cut += sum(len(ids) > MAX_LEN for ids in tokenizer(batch)["input_ids"])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            hidden = model(**enc).last_hidden_state; mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            parts.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    vectors = np.zeros((len(texts), parts[0].shape[1]), "float32"); vectors[order] = np.concatenate(parts)
    del model; torch.cuda.empty_cache()
    return normalize(vectors), cut / len(texts)
''', r'''
from gensim.models import Word2Vec
reps, rep_info = {}, []
def add_rep(name, function, view):
    start = time.perf_counter(); X, extra = function()
    reps[name] = X
    rep_info.append({"representasi": name, "input": view, "dimensi": X.shape[1], "detik": time.perf_counter() - start,
                     "baris_vektor_nol_%": 100 * np.mean(np.asarray(abs(X).sum(1)).ravel() == 0), **extra})

def tfidf_view(view):
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=.5, ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
    return vectorizer.fit_transform(data[view])
print("Representasi disiapkan setelah ablasi preprocessing BoW di 06.1")
''')
md('**6.1 Ablasi preprocessing BoW.** e5 dihitung lebih dulu, karena silhouette di ruang e5 dipakai untuk memilih. TF-IDF + KMeans pada k tengah dari grid, untuk tiga versi teks. Versi dipilih dengan aturan yang sama seperti bagian 07, bukan dari audit.')
code(r'''
E5, E5_CUT = embed(locate("e5", "intfloat/multilingual-e5-base"), ("query: " + data.embed).tolist())
K_REF = K_GRID[len(K_GRID) // 2]; ablation = []
for view in ["normal", "bow", "bow_stem"]:
    X = tfidf_view(view)
    labels = KMeans(K_REF, n_init=3, random_state=SEED).fit_predict(X)
    ablation.append({"view": view, "vocab": X.shape[1], "stability_ARI": stability(X, K_REF), **evaluate(f"ablasi_{view}", labels, X, rep="tfidf", algo="ablasi")})
rows = [r for r in rows if r["algo"] != "ablasi"]
ablation = pd.DataFrame(ablation).drop(columns=["model", "rep", "algo"])
show("Ablasi preprocessing · TF-IDF KMeans k=%d" % K_REF, ablation)
pool = ablation[ablation["max_share_%"] <= 100 * MAX_SHARE].copy()
pool = pool if len(pool) else ablation.copy()
pool["skor_gabungan"] = pool[["silhouette_e5", "stability_ARI"]].rank(pct=True).mean(axis=1)
BOW_VIEW = pool.sort_values(["skor_gabungan", "NPMI"]).iloc[-1]["view"]
print("Versi BoW terpilih (tanpa label):", BOW_VIEW)
''')
md('**6.2 Bangun lima representasi.**')
code(r'''
add_rep("tfidf", lambda: (tfidf_view(BOW_VIEW), {}), BOW_VIEW)
add_rep("lsa", lambda: (normalize(TruncatedSVD(100, random_state=SEED).fit_transform(reps["tfidf"])), {}), BOW_VIEW)
def word2vec():
    tokens = data[BOW_VIEW].str.split().tolist()
    model = Word2Vec(tokens, vector_size=100, window=5, min_count=2, sg=1, epochs=20, workers=1, seed=SEED,
                     hashfxn=lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16))
    vectors = np.array([np.mean([model.wv[w] for w in s if w in model.wv], 0) if any(w in model.wv for w in s)
                        else np.zeros(100) for s in tokens], dtype="float32")
    return normalize(vectors), {"OOV_token_%": 100 * np.mean([w not in model.wv for s in tokens for w in s])}
add_rep("word2vec", word2vec, BOW_VIEW)
def indobert():
    X, cut = embed(locate("indobert", "indobenchmark/indobert-base-p1"), data.embed.str.lower().tolist())
    return X, {"terpotong_%": 100 * cut}
add_rep("indobert_mean", indobert, "embed lowercase")
def e5():
    return E5, {"terpotong_%": 100 * E5_CUT}
add_rep("e5", e5, "query: + embed")
show("Representasi", pd.DataFrame(rep_info))
''')

part('07 · Model: grid KMeans per representasi', 'Setiap representasi dicoba pada semua `K_GRID`. Hasilnya menunjukkan pengaruh representasi saat algoritmanya sama.\n\n' + '**Aturan pemilihan tanpa label.** Cluster raksasa dilarang (`max_share_% <= MAX_SHARE`). Sisanya diurutkan dengan rata-rata peringkat **silhouette di ruang e5** dan **stabilitas**, lalu NPMI dipakai sebagai pemecah seri. Aturan ini berasal dari audit data berita di `03_clustering_review`: Spearman terhadap NMI adalah +0,61 untuk silhouette e5, +0,32 untuk stabilitas, +0,28 untuk NPMI, dan −0,78 untuk ukuran cluster terbesar. Aturan lama (stabilitas, lalu NPMI) memilih Word2Vec dengan NMI 0,116, sedangkan aturan ini memilih e5 dengan NMI 0,197. Karena dirancang dari satu dataset berlabel, aturan ini adalah prior, bukan jaminan untuk data lain.', r'''
for name, X in reps.items():
    for k in K_GRID:
        labels = KMeans(k, n_init=3, random_state=SEED).fit_predict(X)
        evaluate(f"{name}_kmeans", labels, X, rep=name, algo="kmeans", stability_ARI=stability(X, k))
grid = pd.DataFrame(rows)
show("Grid KMeans · tanpa kolom audit", grid.drop(columns=[c for c in grid if c.startswith("audit_")]).sort_values("NPMI", ascending=False).head(15))
''', r'''
pool = grid[grid["max_share_%"] <= 100 * MAX_SHARE].copy()
pool = pool if len(pool) else grid.copy()
pool["skor_gabungan"] = pool[["silhouette_e5", "stability_ARI"]].rank(pct=True).mean(axis=1)
base = pool.sort_values(["skor_gabungan", "NPMI"]).iloc[-1]
REP_STAR, K_STAR = base.rep, int(base.k)
show("5 kandidat teratas menurut aturan", pool.sort_values("skor_gabungan", ascending=False)[["rep", "k", "silhouette_e5", "stability_ARI", "NPMI", "max_share_%", "skor_gabungan"]].head(5))
print(f"Lolos larangan cluster raksasa: {len(pool)} dari {len(grid)} | TERPILIH: {REP_STAR} k={K_STAR}")
''')

part('08 · Model: algoritma lain pada k terpilih', 'Algoritma yang cara kerjanya berbeda, pada k yang sama: topic model (LDA, NMF), hierarkis (Agglomerative Ward), dan berbasis kepadatan (UMAP + HDBSCAN, yang menentukan k sendiri dan boleh membuang noise). Pembanding acak dibuat dengan ukuran cluster yang sama, sebagai batas bawah.', r'''
X_star = reps[REP_STAR]
base_labels = KMeans(K_STAR, n_init=10, random_state=SEED).fit_predict(X_star)
evaluate("final_base_kmeans", base_labels, X_star, rep=REP_STAR, algo="kmeans_n10")
evaluate("acak_ukuran_sama", np.random.default_rng(SEED).permutation(base_labels), X_star, rep=REP_STAR, algo="acak")
counts_bow = CountVectorizer(min_df=MIN_DF, max_df=.5, token_pattern=r"\S+").fit_transform(data[BOW_VIEW])
lda = LatentDirichletAllocation(K_STAR, learning_method="batch", max_iter=30, random_state=SEED).fit_transform(counts_bow)
evaluate("lda_argmax", lda.argmax(1), reps["tfidf"], rep="count", algo="lda")
nmf = NMF(K_STAR, init="nndsvda", max_iter=400, random_state=SEED).fit_transform(reps["tfidf"])
evaluate("nmf_argmax", nmf.argmax(1), reps["tfidf"], rep="tfidf", algo="nmf")
for name in dict.fromkeys(["lsa", "e5", REP_STAR if REP_STAR != "tfidf" else "lsa"]):
    evaluate(f"{name}_agglo_ward", AgglomerativeClustering(K_STAR, linkage="ward").fit_predict(reps[name]), reps[name], rep=name, algo="agglomerative")
''', r'''
import umap
U = umap.UMAP(n_neighbors=30, n_components=10, min_dist=0., metric="cosine", random_state=SEED).fit_transform(E5)
hdb = HDBSCAN(min_cluster_size=max(15, len(data) // 100), min_samples=5).fit_predict(U)
evaluate("e5_umap_hdbscan", hdb, E5, rep="e5_umap", algo="hdbscan")
evaluate("e5_umap_kmeans", KMeans(K_STAR, n_init=3, random_state=SEED).fit_predict(U), E5, rep="e5_umap", algo="kmeans")
board = pd.DataFrame(rows)
show("Perbandingan algoritma · tanpa audit", board[board.algo != "kmeans"].drop(columns=[c for c in board if c.startswith("audit_")]))
''')

part('09 · Post-processing', '''Diterapkan pada model terpilih, dan setiap langkah dinilai ulang.
- **Konsensus:** 10 KMeans dengan seed berbeda digabung lewat matriks co-association, lalu dipotong dengan average linkage. Tujuannya mengurangi ketergantungan pada satu seed.
- **Gabung cluster kecil:** cluster di bawah `MIN_CLUSTER_SHARE` dipindah ke centroid terdekat, karena cluster puluhan baris sulit dijelaskan saat presentasi.
- **Vektor nol:** di representasi berbasis kata, teks tanpa kata yang dikenal menjadi vektor nol, lalu semuanya jatuh ke satu cluster. Teks seperti ini diarahkan ke cluster terdekat di ruang e5.
- **Konsistensi duplikat:** teks yang identik setelah normalisasi dipaksa ke cluster mayoritasnya.
- **Noise HDBSCAN:** titik noise dipindah ke cluster terdekat, supaya semua baris punya cluster.''', r'''
def centroid_assign(X, labels, members):
    X = X.toarray() if hasattr(X, "toarray") else X
    keep = [c for c in np.unique(labels) if c >= 0 and c not in members]
    centers = normalize(np.vstack([X[labels == c].mean(0) for c in keep]))
    moved = labels.copy(); mask = np.isin(labels, list(members)) | (labels < 0)
    moved[mask] = np.array(keep)[(normalize(X[mask]) @ centers.T).argmax(1)]
    return moved

runs = [KMeans(K_STAR, n_init=1, random_state=SEED + s).fit_predict(X_star) for s in range(10)]
onehot = np.hstack([np.eye(K_STAR, dtype="float32")[r] for r in runs])
coassoc = onehot @ onehot.T / len(runs); del onehot
consensus = AgglomerativeClustering(K_STAR, metric="precomputed", linkage="average").fit_predict(1 - coassoc); del coassoc
evaluate("post_konsensus", consensus, X_star, rep=REP_STAR, algo="post")
sizes = pd.Series(base_labels).value_counts(normalize=True)
merged = centroid_assign(X_star, base_labels, set(sizes[sizes < MIN_CLUSTER_SHARE].index)) if (sizes < MIN_CLUSTER_SHARE).any() else base_labels
evaluate("post_gabung_kecil", merged, X_star, rep=REP_STAR, algo="post")
zero = np.asarray(abs(X_star).sum(1)).ravel() == 0
if zero.any():  # vektor nol (tidak ada kata yang dikenal) diarahkan lewat e5, bukan ditumpuk ke satu cluster
    known = np.unique(merged[~zero])
    e5_centers = normalize(np.vstack([E5[(merged == c) & ~zero].mean(0) for c in known]))
    merged = merged.copy(); merged[zero] = known[(E5[zero] @ e5_centers.T).argmax(1)]
    evaluate("post_vektor_nol_lewat_e5", merged, X_star, rep=REP_STAR, algo="post")
print("Baris vektor nol diarahkan lewat e5:", int(zero.sum()))
dup_key = data.normal.str.replace(r"\s+", " ", regex=True).str.strip()
consistent = pd.Series(merged).groupby(dup_key.to_numpy()).transform(lambda s: s.mode().iloc[0]).to_numpy()
evaluate("post_gabung_kecil+duplikat", consistent, X_star, rep=REP_STAR, algo="post")
evaluate("post_hdbscan_tanpa_noise", centroid_assign(U, hdb, set()), E5, rep="e5_umap", algo="post")
print("Cluster kecil dipindah:", int((sizes < MIN_CLUSTER_SHARE).sum()), "| label berubah karena duplikat:", int((consistent != merged).sum()))
''', r'''
board = pd.DataFrame(rows)
post = board[board.model.isin(["final_base_kmeans", "post_konsensus", "post_gabung_kecil", "post_vektor_nol_lewat_e5", "post_gabung_kecil+duplikat"])].set_index("model")
final_name = "post_gabung_kecil+duplikat"
if post.loc["post_konsensus", "NPMI"] > post.loc[final_name, "NPMI"] + .005: final_name = "post_konsensus"
final = {"post_konsensus": consensus, "post_gabung_kecil+duplikat": consistent}[final_name]
show("Post-processing · tanpa audit", post.drop(columns=[c for c in post if c.startswith("audit_")]))
print("FINAL (dipilih tanpa label):", final_name, "| rep:", REP_STAR, "| k:", len(np.unique(final)))
''')

part('10 · Evaluasi & kelemahan', 'Mulai dari sini kolom audit dibuka. Keputusan di atas tidak diubah. Bagian ini menjawab tiga pertanyaan: seberapa jauh pilihan tanpa label dari pilihan terbaik, kriteria tanpa label mana yang bisa dipercaya, dan cluster mana yang lemah.', r'''
board = pd.DataFrame(rows)
audit_cols = [c for c in board if c.startswith("audit_")]
cols = ["model", "rep", "algo", "k", "stability_ARI", "NPMI", "diversity", "silhouette_own", "silhouette_e5", "max_share_%", "noise_%"] + audit_cols
show("10.1 Semua kandidat", board[[c for c in cols if c in board]].sort_values(audit_cols[0] if audit_cols else "NPMI", ascending=False))
if AUDIT_COL:
    ranked = board[~board.algo.isin(["acak"])].sort_values("audit_NMI", ascending=False).reset_index(drop=True)
    chosen = board.loc[board.model.eq(final_name), "audit_NMI"].iloc[0]
    print(f"Pilihan tanpa label: NMI {chosen:.3f}, peringkat {int((ranked.audit_NMI > chosen).sum()) + 1} dari {len(ranked)} | "
          f"terbaik dengan melihat label: {ranked.model[0]} ({ranked.rep[0]}, k={ranked.k[0]}) NMI {ranked.audit_NMI[0]:.3f} | acak: "
          f"{board.loc[board.model.eq('acak_ukuran_sama'), 'audit_NMI'].iloc[0]:.3f}")
''', r'''
if AUDIT_COL:
    candidates = board[board.algo.eq("kmeans")]
    trust = pd.DataFrame({m: spearmanr(candidates[m], candidates.audit_NMI, nan_policy="omit") for m in
        ["stability_ARI", "NPMI", "diversity", "silhouette_own", "silhouette_e5", "max_share_%", "k"] if m in candidates}, index=["spearman", "p"]).T
    show("10.2 Kriteria tanpa label vs NMI audit · grid KMeans, positif = bisa dipercaya", trust.sort_values("spearman", ascending=False))
    best_rep = candidates.groupby("rep").audit_NMI.max().sort_values(ascending=False)
    show("NMI terbaik per representasi · apa pengaruh representasi", best_rep)
''', r'''
labels = final; X_final = X_star.toarray() if hasattr(X_star, "toarray") else X_star
keys = keywords(labels); centers = normalize(np.vstack([X_final[labels == c].mean(0) for c in sorted(keys)]))
cosine = (normalize(X_final) * centers[np.searchsorted(sorted(keys), labels)]).sum(1)
_, boot = stability(X_star, len(keys), return_runs=True)
def cluster_stability(c):
    member = labels == c
    return np.mean([max((member & (r == j)).sum() / (member | (r == j)).sum() for j in np.unique(r)) for r in boot])
cards = pd.DataFrame([{"cluster": c, "n": int((labels == c).sum()), "%": 100 * (labels == c).mean(), "kata_kunci": ", ".join(keys[c][:8]),
    "NPMI": npmi(keys[c]), "stabilitas_jaccard": cluster_stability(c), "kemiripan_ke_pusat": cosine[labels == c].mean(),
    "contoh": " | ".join(data.loc[labels == c].assign(s=cosine[labels == c]).nlargest(2, "s")[TEXT_COL].str.slice(0, 70))} for c in sorted(keys)])
if AUDIT_COL:
    truth = data["_audit"].to_numpy()
    cards["audit_mayoritas"] = [pd.Series(truth[labels == c]).mode()[0] for c in cards.cluster]
    cards["audit_purity"] = [pd.Series(truth[labels == c]).value_counts(normalize=True).iloc[0] for c in cards.cluster]
show("10.3 Kartu cluster final · urut dari paling lemah (stabilitas)", cards.sort_values("stabilitas_jaccard"))
''', r'''
if AUDIT_COL:
    table = pd.crosstab(pd.Series(truth, name="kategori_audit"), pd.Series(labels, name="cluster"), normalize="index") * 100
    show("10.4 Kategori audit tersebar ke cluster mana · % baris kategori", table)
    split = (table.apply(lambda r: (r.sort_values(ascending=False).cumsum() < 80).sum() + 1, axis=1)).rename("cluster_untuk_80%")
    show("Kategori yang dipecah · butuh banyak cluster untuk 80% barisnya", split.sort_values(ascending=False))
length = pd.cut(data.bow.str.split().str.len(), [-1, 0, 3, 6, 12, np.inf], labels=["kosong", "1-3", "4-6", "7-12", ">12"])
slice_rows = []
for value, ids in pd.Series(np.arange(len(data))).groupby(length.to_numpy(), observed=True):
    ids = ids.to_numpy(); row = {"panjang_bow": value, "n": len(ids), "kemiripan_ke_pusat": cosine[ids].mean()}
    if AUDIT_COL: row.update(NMI_slice=normalized_mutual_info_score(truth[ids], labels[ids]),
        salah_mayoritas_cluster_pct=100 * np.mean(truth[ids] != cards.set_index("cluster").audit_mayoritas.reindex(labels[ids]).to_numpy()))
    slice_rows.append(row)
show("10.5 Kelemahan menurut panjang teks setelah preprocessing", pd.DataFrame(slice_rows))
''', r'''
analysis = data[["row_id", TEXT_COL]].assign(cluster=labels, kemiripan_ke_pusat=cosine)
show("10.6 Paling jauh dari pusat cluster · kandidat salah tempat atau outlier", analysis.nsmallest(8, "kemiripan_ke_pusat"))
if AUDIT_COL:
    analysis["audit"] = truth
    analysis["audit_mayoritas_cluster"] = cards.set_index("cluster").audit_mayoritas.reindex(labels).to_numpy()
    wrong = analysis[analysis.audit != analysis.audit_mayoritas_cluster]
    show("10.7 Yakin tapi beda dari mayoritas cluster · sering bukan salah model, kategori aslinya tumpang tindih",
         wrong.nlargest(8, "kemiripan_ke_pusat"))
weakest = cards.sort_values("stabilitas_jaccard").iloc[0]
print(f"Cluster paling tidak stabil: {int(weakest.cluster)} ({weakest.kata_kunci}) | Jaccard {weakest.stabilitas_jaccard:.2f} | n={int(weakest.n)}")
''')

part('11 · Keluaran', 'File `clusters.csv` berisi satu cluster untuk setiap baris, yang menjadi bentuk submission clustering. Disimpan juga papan kandidat, kartu cluster, dan manifest run.', r'''
analysis.drop(columns=[c for c in ["audit", "audit_mayoritas_cluster"] if c in analysis]).assign(
    kata_kunci=[", ".join(keys[c][:5]) for c in labels]).to_csv(OUT / "clusters.csv", index=False)
board.to_csv(OUT / "kandidat.csv", index=False); cards.to_csv(OUT / "kartu_cluster.csv", index=False)
ablation.to_csv(OUT / "ablasi_preprocessing.csv", index=False); pd.DataFrame(rep_info).to_csv(OUT / "representasi.csv", index=False)
import importlib.metadata as metadata
manifest = {"dataset": DATASET, "rows": len(data), "raw_rows": len(raw), "seed": SEED, "bow_view": BOW_VIEW, "rep": REP_STAR,
    "k": int(len(np.unique(final))), "final": final_name, "domain_stopwords": sorted(DOMAIN_STOP),
    "final_scores": {k: v for k, v in board.loc[board.model.eq(final_name)].iloc[0].items() if isinstance(v, (int, float, np.floating))},
    "seconds": time.perf_counter() - RUN_START, "gpu": torch.cuda.get_device_name(0),
    "versions": {p: metadata.version(p) for p in ["torch", "transformers", "scikit-learn", "umap-learn", "gensim", "pandas"]}}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
print(json.dumps(manifest, indent=2, default=float))
''')

md(r'''
## Sumber & batas interpretasi
- Fungsi preprocessing disalin dari `preprocessing_universal/stepwise.py`. Daftar stopword tambahan dan stopword domain bergantung pada data, jadi periksa tabel 4.1 sebelum dipakai.
- [multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-base) memakai awalan `query: `. IndoBERT tidak dilatih untuk kemiripan kalimat, jadi mean pooling-nya hanya pembanding.
- NPMI mengikuti [Röder dkk. 2015](https://dl.acm.org/doi/10.1145/2684822.2685324). c-TF-IDF mengikuti [BERTopic](https://maartengr.github.io/BERTopic/).
- Kategori berita tidak sama dengan topik "benar". Satu kategori bisa berisi beberapa topik, jadi NMI audit adalah batas kasar, bukan kebenaran mutlak.
- Satu seed dan satu sampel 5.000 baris. Stabilitas mengukur variasi KMeans, bukan variasi sampel data.
''')

for cell in CELLS:
    if cell.cell_type == 'code': ast.parse(cell.source)
config = next(c for c in CELLS if c.cell_type == 'code')
config.metadata['tags'] = ['parameters']
notebook = nb.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "pilot": {"task": "clustering", "preprocessing_source_sha256": hashlib.sha256(u.encode()).hexdigest()}})
nb.validate(notebook); nb.write(notebook, DEST)

kernel = Path(tempfile.gettempdir()) / 'gemastik_clustering_review_kernel'; kernel.mkdir(exist_ok=True)
payload = base64.b64encode(nb.writes(notebook).encode()).decode()
runner = f'''import base64, pathlib, os, traceback
import papermill as pm
os.environ['PYTHONHASHSEED'] = '42'
source = pathlib.Path('/kaggle/working/_source.ipynb')
source.write_bytes(base64.b64decode({payload!r}))
for dataset, name in [('news', '03_clustering_review.ipynb'), ('mbg', '03_clustering_review_mbg.ipynb')]:
    try:
        pm.execute_notebook(str(source), '/kaggle/working/' + name, parameters={{'DATASET': dataset}},
                            kernel_name='python3', cwd='/kaggle/working', progress_bar=False, log_output=True)
    except Exception:
        traceback.print_exc()
source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)], metadata=notebook.metadata), kernel / 'clustering.ipynb')
meta = {"id": "ivanwllm/gemastik-clustering-review", "title": "Gemastik Clustering Review", "code_file": "clustering.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True, "dataset_sources": ["ibamibrahim/indonesian-news-title", "dzikriraihan/indonesian-youtube-comments-mbg-program"],
        "kernel_sources": ["ivanwllm/gemastik-nlp-00-setup-offline"], "competition_sources": [], "model_sources": []}
(kernel / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('Wrote', DEST, '| kernel', kernel, '| cells', len(CELLS), '| code lines', sum(len(c.source.splitlines()) for c in CELLS if c.cell_type == 'code'))
